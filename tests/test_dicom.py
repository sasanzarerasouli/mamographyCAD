"""DICOM decoding: MONOCHROME1/2, 8/16 bit, signed data, spacing."""
import numpy as np
import pytest

pydicom = pytest.importorskip("pydicom")

from src.io.dicom_handler import DICOMHandler


def _write_dicom(path, array, photometric="MONOCHROME2", bits=16, spacing=(0.07, 0.07),
                 window=None, signed=False):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = generate_uid()
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = file_meta
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = "TEST001"
    ds.PatientName = "Test^Patient"
    ds.Modality = "MG"
    ds.StudyDate = "20240101"
    ds.ViewPosition = "MLO"
    ds.ImageLaterality = "L"
    ds.PixelSpacing = list(spacing)
    ds.Rows, ds.Columns = array.shape
    ds.BitsAllocated = bits
    ds.BitsStored = bits if bits <= 16 else 16
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = 1 if signed else 0
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    if window:
        ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = array.tobytes()
    ds.save_as(path, enforce_file_format=True)
    return path


def test_monochrome2_16bit(tmp_path):
    array = np.tile(np.linspace(0, 4095, 64, dtype=np.uint16), (64, 1))
    path = _write_dicom(str(tmp_path / "m2.dcm"), array, "MONOCHROME2", 16)
    gray, metadata = DICOMHandler().load_grayscale(path)
    assert gray is not None and gray.dtype == np.uint8
    assert metadata["image"]["photometric_interpretation"] == "MONOCHROME2"
    assert gray[32, -1] > gray[32, 0], "bright end must stay bright"


def test_monochrome1_is_inverted(tmp_path):
    array = np.tile(np.linspace(0, 4095, 64, dtype=np.uint16), (64, 1))
    path = _write_dicom(str(tmp_path / "m1.dcm"), array, "MONOCHROME1", 16)
    gray, metadata = DICOMHandler().load_grayscale(path)
    assert metadata["image"]["monochrome1_inverted"] is True
    assert gray[32, 0] > gray[32, -1], "MONOCHROME1 must be inverted for display"


def test_8bit_and_bits_stored(tmp_path):
    array = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    path = _write_dicom(str(tmp_path / "m8.dcm"), array, "MONOCHROME2", 8)
    gray, metadata = DICOMHandler().load_grayscale(path)
    assert gray is not None
    assert metadata["image"]["bits_allocated"] == 8


def test_signed_pixel_data(tmp_path):
    array = np.tile(np.linspace(-1000, 1000, 64), (64, 1)).astype(np.int16)
    path = _write_dicom(str(tmp_path / "m1signed.dcm"), array, "MONOCHROME2", 16, signed=True)
    gray, metadata = DICOMHandler().load_grayscale(path)
    assert gray is not None
    assert metadata["image"]["pixel_representation"] == 1
    assert gray[32, -1] > gray[32, 0]


def test_pixel_spacing_is_read(tmp_path):
    array = np.zeros((64, 64), np.uint16)
    path = _write_dicom(str(tmp_path / "sp.dcm"), array, spacing=(0.05, 0.1))
    handler = DICOMHandler()
    _, metadata = handler.load_grayscale(path)
    row, col = handler.get_pixel_spacing(metadata)
    assert (row, col) == (0.05, 0.1)
    assert handler.is_spacing_known(metadata)
    assert handler.get_view_position(metadata) == "MLO"


def test_fallback_for_non_dicom(tmp_path):
    import cv2

    path = str(tmp_path / "plain.png")
    cv2.imwrite(path, np.full((40, 40), 120, np.uint8))
    gray, metadata = DICOMHandler().load_grayscale(path)
    assert gray is not None and metadata["source"] == "standard"
