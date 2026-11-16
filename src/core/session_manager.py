"""Automatic session recovery and persistence."""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import Config

logger = logging.getLogger("MammographyCAD.session")


class SessionManager:
    """Persist and restore application session state."""

    SESSION_FILE = os.path.join(Config.SESSION_DIR, "last_session.json")

    def save(
        self,
        image_path: Optional[str],
        ignored_ids: List[int],
        zoom_level: float,
        pan_offset: List[float],
        analysis_dict: Optional[Dict] = None,
        annotations: Optional[List] = None,
    ) -> None:
        session = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "ignored_ids": ignored_ids,
            "zoom_level": zoom_level,
            "pan_offset": pan_offset,
            "analysis": analysis_dict,
            "annotations": annotations or [],
        }
        try:
            with open(self.SESSION_FILE, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, default=str)
            logger.debug("Session saved")
        except OSError as exc:
            logger.error("Failed to save session: %s", exc)

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.SESSION_FILE):
            return None
        try:
            with open(self.SESSION_FILE, "r", encoding="utf-8") as f:
                session = json.load(f)
            logger.info("Session restored from %s", session.get("timestamp"))
            return session
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load session: %s", exc)
            return None

    def clear(self) -> None:
        if os.path.exists(self.SESSION_FILE):
            os.remove(self.SESSION_FILE)
