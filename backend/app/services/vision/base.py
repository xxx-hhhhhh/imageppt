from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class VisionProvider(ABC):
    name = "base"

    @property
    def provider_name(self) -> str:
        return self.name

    @property
    def model_name(self) -> str | None:
        return getattr(self, "model", None)

    @abstractmethod
    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict[str, Any]:
        raise NotImplementedError

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": False}

    def test_connection(self) -> dict[str, Any]:
        return {"success": False, "provider": self.name, "model": self.model_name, "message": "Provider 未配置"}


class NullVisionProvider(VisionProvider):
    name = "none"

    def analyze_scene(self, image_path: Path, context: dict | None = None, mode: str = "standard") -> dict[str, Any]:
        return {"provider": self.name, "page": {}, "regions": [], "elements": [], "groups": [], "relations": [], "repeatedComponents": [], "layers": [], "confidence": 0.0}

    def critique_reconstruction(self, original_path: Path, reconstructed_path: Path, scene: dict) -> dict[str, Any]:
        return {"provider": self.name, "overallAssessment": {"layout": 0.0, "typography": 0.0, "graphics": 0.0}, "issues": []}

    def status(self) -> dict[str, Any]:
        return {"provider": self.name, "configured": False, "model": getattr(self, "model", None)}
