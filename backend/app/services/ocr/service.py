from __future__ import annotations

from pathlib import Path

from app.services.ocr.provider import OCRResult, PaddleOCRProvider, RapidOCRProvider, create_ocr_provider


class OCRService:
    def __init__(self, preferred: str = "auto") -> None:
        self.preferred = preferred
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
            failed_name = self.provider.name
            self.warnings.append(f"{failed_name} inference failed: {exc}")
            if isinstance(self.provider, PaddleOCRProvider):
                try:
                    self.provider = RapidOCRProvider()
                    results = self.provider.recognize(image_path)
                    self.warnings.append("PaddleOCR unavailable during inference, using RapidOCR fallback")
                    return results
                except Exception as fallback_exc:
                    self.warnings.append(f"RapidOCR fallback failed: {fallback_exc}")
            self.warnings.append("OCR failed and returned no text objects")
            return []
