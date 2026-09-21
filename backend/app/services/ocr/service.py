from __future__ import annotations

from pathlib import Path

from app.services.ocr.provider import OCRResult, create_ocr_provider


class OCRService:
    def __init__(self, preferred: str = "auto") -> None:
        self.provider, self.warnings = create_ocr_provider(preferred)

    @property
    def provider_name(self) -> str:
        return self.provider.name

    def recognize(self, image_path: Path) -> list[OCRResult]:
        try:
            return self.provider.recognize(image_path)
        except NotImplementedError:
            self.warnings.append("No OCR engine is installed; upload and editing still work, but no text objects were detected.")
            return []
        except Exception as exc:
            self.warnings.append(f"OCR failed and returned no text objects: {exc}")
            return []

