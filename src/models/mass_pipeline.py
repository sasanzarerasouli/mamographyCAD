"""Classical mass localization and original-grid ROI segmentation.

Scores are research heuristics, not malignancy probabilities. Pixel-scale
parameters are starting points and require dataset-specific validation.
"""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.segmentation import watershed

from config import Config
from ..core.detection_types import Detection, DetectionType


def normalize_response(response, valid):
    values = response[valid]
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = max(1.4826 * mad, float(np.quantile(np.abs(values - median), .95)) / 4, 1e-6)
    return np.clip((response - median) / sigma, 0, 8).astype(np.float32)


def gaussian(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)


def response_maps(image, valid):
    """Return independent peak maps, common-scale fusion, scale and polarity."""
    scale_maps = []
    maps = np.zeros((5, *image.shape), np.float32)
    fused = np.zeros_like(image)
    scales = np.zeros_like(image)
    polarity = np.zeros(image.shape, np.int8)
    support = np.zeros(image.shape, np.uint8)
    weights = np.asarray(Config.MASS_RESPONSE_WEIGHTS, np.float32)
    if weights.shape != (5,) or np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError('MASS_RESPONSE_WEIGHTS must contain five finite nonnegative values with a positive sum')
    for s in Config.MASS_SCALES:
        inner = gaussian(image, s)
        outer = gaussian(image, s * Config.MASS_DOG_RATIO)
        hxx = cv2.Sobel(inner, -1, 2, 0, scale=s*s/4)
        hyy = cv2.Sobel(inner, -1, 0, 2, scale=s*s/4)
        hxy = cv2.Sobel(inner, -1, 1, 1, scale=s*s/4)
        trace = (hxx + hyy) / 2
        spread = np.hypot((hxx - hyy) / 2, hxy)
        k = max(1, round(s))
        center = cv2.boxFilter(image, -1, (2*k+1,)*2)
        total = cv2.boxFilter(image, -1, (4*k+1,)*2)
        fraction = ((2*k+1)/(4*k+1))**2
        annulus = (total - fraction * center) / (1 - fraction)
        local_sd = np.sqrt(np.maximum(cv2.boxFilter(image**2, -1, (4*k+1,)*2) - total**2, 0))
        contrast = (center - annulus) / (local_sd + .02)
        # Evaluate both polarities independently; do not mix bright and dark evidence.
        best = np.zeros_like(image)
        best_sign = np.zeros(image.shape, np.int8)
        best_support = np.zeros(image.shape, np.uint8)
        kernel = np.ones((2*round(1.6*s)+1,)*2, np.uint8)
        for sign in (1, -1):
            signed = image * sign
            compact = np.maximum(-sign * trace - spread, 0)
            raw = (sign*(inner-outer), -2*sign*trace, compact, sign*contrast,
                   signed - cv2.morphologyEx(signed, cv2.MORPH_OPEN, kernel))
            score = np.zeros_like(image)
            count = np.zeros_like(support)
            for i, response in enumerate(raw):
                z = normalize_response(response, valid)
                np.maximum(maps[i], z, out=maps[i])
                score += weights[i] * z / weights.sum()
                count += z >= Config.MASS_SUPPORT_Z
            better_scale = valid & (score > best)
            best[better_scale] = score[better_scale]
            best_sign[better_scale] = sign
            best_support[better_scale] = count[better_scale]
            better = valid & (score > fused)
            fused[better], scales[better], polarity[better] = score[better], s, sign
            support[better] = count[better]
        scale_maps.append((s, best, best_sign, best_support))
    maps[:, ~valid] = 0
    return maps, fused, scales, polarity, support, scale_maps


def associated(mask, y, x, radius):
    labels, _ = ndi.label(mask, structure=np.ones((3, 3)))
    label = labels[y, x]
    if not label:
        iy, ix = np.nonzero(labels)
        if not len(ix):
            return np.zeros_like(mask, dtype=bool)
        distance = (iy-y)**2 + (ix-x)**2
        j = distance.argmin()
        if distance[j] > radius**2:
            return np.zeros_like(mask, dtype=bool)
        label = labels[iy[j], ix[j]]
    return labels == label


def hypotheses(image, valid, y, x, scale):
    yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
    distance = np.hypot(yy-y, xx-x)
    smooth = gaussian(image, 1)
    gradient = np.hypot(ndi.sobel(smooth, axis=0), ndi.sobel(smooth, axis=1))
    data = smooth[valid]
    k = Config.MASS_FCM_CLUSTERS
    centers = np.quantile(data, np.linspace(.1, .9, k))
    for _ in range(Config.MASS_FCM_ITERATIONS):
        membership = 1 / np.maximum(np.abs(data[:, None] - centers), 1e-5)**2
        membership /= membership.sum(axis=1, keepdims=True)
        weights = membership**2
        updated = (data @ weights) / weights.sum(axis=0)
        if np.max(np.abs(updated - centers)) < 1e-4:
            break
        centers = updated
    labels = np.full(image.shape, -1)
    labels[valid] = membership.argmax(axis=1)
    core = valid & (distance <= max(2, .65*scale))
    ring = valid & (distance >= 1.8*scale)
    options = []
    for i in range(k):
        component = associated(labels == i, y, x, .55*scale)
        if not component.any():
            continue
        core_fraction = np.mean(component[core])
        ring_fraction = np.mean(component[ring]) if ring.any() else 0
        contrast = np.median(smooth[component]) - (np.median(smooth[ring]) if ring.any() else np.median(data))
        score = core_fraction - ring_fraction + .35 * np.clip(contrast/.1, -1, 1)
        options.append((score, component, float(membership[component[valid], i].mean())))
    _, fcm, member = max(options, key=lambda row: row[0]) if options else (0, np.zeros_like(valid), 0.)

    # Marker-controlled watershed finds a connected boundary on the gradient
    # surface; distance is used only to place seeds, never to generate a mask.
    markers = np.zeros(image.shape, np.int32)
    markers[distance >= Config.MASS_BOUNDARY_OUTER * scale] = 2
    markers[[0, -1], :] = 2
    markers[:, [0, -1]] = 2
    markers[~valid] = 2
    markers[valid & (distance <= Config.MASS_BOUNDARY_INNER * scale)] = 1
    boundary = (watershed(gradient, markers) == 1) & valid

    local = gaussian(smooth, max(2., Config.MASS_ADAPTIVE_SIGMA * scale))
    adaptive = associated(valid & (smooth > local + Config.MASS_ADAPTIVE_OFFSET), y, x, .55*scale)
    return (fcm, boundary, adaptive), member, gradient, ring


def detect_masses(image, breast_mask, debug=None):
    valid = breast_mask > 0
    if not valid.any():
        return []
    lo, hi = np.percentile(image[valid], (1, 99.8)).astype(np.float32)
    if hi <= lo:
        return []
    norm = np.clip((image.astype(np.float32) - lo)/(hi-lo), 0, 1)
    # Localization pyramid only; segmentation and masks retain original pixels.
    h, w = norm.shape
    ratio = min(1., Config.MASS_LOCALIZATION_MAX_SIDE / max(h, w))
    size = (max(1, round(w*ratio)), max(1, round(h*ratio)))
    small = cv2.resize(norm, size, interpolation=cv2.INTER_AREA) if ratio < 1 else norm.copy()
    tissue = cv2.resize(valid.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    if not tissue.any():
        return []
    # Extend tissue values at the breast border to reduce artificial edge peaks.
    nearest = ndi.distance_transform_edt(~tissue, return_distances=False, return_indices=True)
    small[~tissue] = small[nearest[0][~tissue], nearest[1][~tissue]]
    maps, fused, scales, polarity, support, scale_maps = response_maps(small, tissue)
    distance = ndi.distance_transform_edt(tissue)
    peak_score = fused * (.55 + .45 * np.minimum(distance / np.maximum(scales, 1), 1))
    if debug is not None:
        debug.update(maps=maps, fused=fused, peak_score=peak_score, scale=scales, polarity=polarity,
                     support=support, original_shape=(h, w))
    candidates = []
    for s, scores, signs, counts in scale_maps:
        rank = scores * (.55 + .45 * np.minimum(distance / s, 1))
        peaks = (rank >= Config.MASS_PEAK_Z) & (rank == cv2.dilate(rank.astype(np.float32), np.ones((5, 5), np.uint8)))
        ys, xs = np.nonzero(peaks)
        chosen = []
        for i in np.argsort(-rank[ys, xs], kind='stable'):
            sy, sx = ys[i], xs[i]
            if any(np.hypot(sx-px, sy-py) < Config.MASS_NMS_SCALE*s for py, px in chosen):
                continue
            chosen.append((sy,sx))
            candidates.append((rank[sy,sx], sy,sx,float(s), int(signs[sy,sx]),
                               int(counts[sy,sx]), float(scores[sy,sx])))
            if len(chosen) >= Config.MASS_MAX_ROIS_PER_SCALE:
                break
    candidates.sort(reverse=True)
    detections = []
    for rank, sy, sx, s, sign, count, raw_score in candidates:
        scale = s / ratio
        y = min(h-1, int((sy+.5)*h/size[1]))
        x = min(w-1, int((sx+.5)*w/size[0]))
        r = round(Config.MASS_ROI_SCALE * scale)
        x0, y0, x1, y1 = max(0,x-r), max(0,y-r), min(w,x+r+1), min(h,y+r+1)
        roi = norm[y0:y1, x0:x1] * sign
        inside = valid[y0:y1, x0:x1]
        if not inside[y-y0, x-x0]:
            continue
        masks, member, grad, ring = hypotheses(roi, inside, y-y0, x-x0, scale)
        votes = np.sum(masks, axis=0)
        mask = associated(votes >= 2, y-y0, x-x0, .55*scale)
        holes = ndi.binary_fill_holes(mask) & ~mask & inside
        labels, _ = ndi.label(holes)
        sizes = np.bincount(labels.ravel())
        mask |= holes & (sizes[labels] <= Config.MASS_MAX_HOLE_PIXELS)
        agreement = float(np.count_nonzero(votes == 3) / max(1, np.count_nonzero(votes)))
        area = int(mask.sum())
        enough = Config.MASS_MIN_AREA <= area <= min(.65*inside.sum(), np.pi*(2.5*scale)**2)
        boundary = mask & ~ndi.binary_erosion(mask)
        prominence = float(np.median(roi[mask]) - np.median(roi[ring])) if area and ring.any() else 0.
        edge = float(np.mean(grad[boundary])) if boundary.any() else 0.
        score = float(np.clip(.40*rank/8 + .12*count/5
                              + .30*np.clip(prominence/.32, 0, 1) + .08*np.clip(edge/.5,0,1) + .10*agreement, 0, 1))
        features = dict(source='classical', label='Mass candidate', confidence_kind='uncalibrated heuristic',
                        scale_px=scale, radius=scale, polarity=sign,
                        detector_support=count, fused_response=raw_score,
                        local_contrast=prominence, method_agreement=agreement, fcm_membership=member,
                        segmentation_confidence=agreement if enough else 0., boundary_gradient=edge,
                        needs_review=not enough or agreement < Config.MASS_REVIEW_AGREEMENT)
        bbox, full, contour = None, None, None
        if enough:
            patch = mask.astype(np.uint8)*255
            bx, by, bw, bh = cv2.boundingRect(patch)
            bbox = (x0+bx, y0+by, bw, bh)
            # Avoid duplicate masks at nearly identical locations across scales.
            duplicate = False
            for previous in detections:
                if previous.mask is not None:
                    intersection = np.count_nonzero(mask & (previous.mask[y0:y1, x0:x1] > 0))
                    if intersection / max(1, area+previous.features['area']-intersection) > .4:
                        duplicate = True
                        break
            if duplicate:
                continue
            full = np.zeros((h, w), np.uint8)
            full[y0:y1, x0:x1] = patch
            contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE, offset=(x0,y0))
            contour = max(contours, key=cv2.contourArea)
            props = regionprops(mask.astype(np.uint8))[0]
            p = float(cv2.arcLength(contour, True))
            centroid_y, centroid_x = props.centroid
            coords = contour.reshape(-1, 2)
            radii = np.hypot(coords[:,0] - x0 - centroid_x, coords[:,1] - y0 - centroid_y)
            hull_p = cv2.arcLength(cv2.convexHull(contour), True)
            features.update(area=area, perimeter=p, convexity=float(hull_p/max(1,p)),
                            radial_roughness=float(radii.std()/max(1,radii.mean())), equivalent_diameter_px=float(props.equivalent_diameter_area),
                            circularity=float(4*np.pi*area/max(1,p*p)), solidity=float(props.solidity),
                            aspect_ratio=float(props.axis_major_length/max(1,props.axis_minor_length)),
                            internal_heterogeneity=float(roi[mask].std()))
        detection = Detection(DetectionType.TUMOR_SUSPICIOUS, (x,y), bbox, full, score,
                              features, contour=contour)
        detection.segmentation_hypotheses = tuple(m.astype(np.uint8) for m in masks)
        detection.segmentation_origin = (x0,y0)
        detections.append(detection)
    detections.sort(key=lambda d: (d.mask is not None, d.confidence), reverse=True)
    return detections[:Config.MASS_MAX_FINDINGS]
