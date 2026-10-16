import cv2
import numpy as np
from app.services.ocr.column_recovery import recover_columns
from app.services.ocr.provider import OCRResult
from app.utils import image_io


def test_uncertain_column_is_split_and_mapped_only_after_real_provider_accepts(tmp_path):
    image=np.full((100,70,3),250,np.uint8)
    cv2.rectangle(image,(22,20),(34,32),(30,30,30),-1)
    cv2.rectangle(image,(22,48),(34,60),(30,30,30),-1)
    path=tmp_path/'column.png'
    image_io.imwrite(path,image)
    original=OCRResult('uncertain',[20,18,38,63],.4,{})
    class Provider:
        calls=0
        def recognize(self,path):
            self.calls+=1
            # Explicit provider fixture, not real OCR evidence.
            return [OCRResult(f'label{self.calls}',[52,44,100,88],.98,{})]
    provider=Provider()
    rows=recover_columns(path,[original],provider)
    assert len(rows)==2 and provider.calls==2
    assert [r.text for r in rows]==['label1','label2']
    assert rows[0].bbox[1]<rows[1].bbox[1]
    assert all(r.style['ocrRecoveryPolicy']=='bounded_4x_vertical_column_reread' for r in rows)


def test_low_confidence_or_exception_preserves_original_column(tmp_path):
    image=np.full((100,70,3),250,np.uint8)
    image[20:33,22:35]=30
    image[48:61,22:35]=30
    path=tmp_path/'column.png'
    image_io.imwrite(path,image)
    original=OCRResult('uncertain',[20,18,38,63],.4,{})
    class Provider:
        def recognize(self,path):
            return [OCRResult('guess',[52,44,100,88],.6,{})]
    assert recover_columns(path,[original],Provider())==[original]
    class Failed:
        def recognize(self,path):
            raise RuntimeError('model unavailable')
    assert recover_columns(path,[original],Failed())==[original]


def test_optional_recovery_failure_does_not_discard_successful_ocr(monkeypatch, tmp_path):
    import app.services.ocr.column_recovery as module
    from app.services.ocr.service import OCRService

    original=[OCRResult('original text',[10,10,90,30],.98,{})]

    class Provider:
        name='test fixture'

        def recognize(self,path):
            return original

    def fail(*args):
        raise ValueError('optional crop preparation failed')

    monkeypatch.setattr(module,'recover_columns',fail)
    service=OCRService.__new__(OCRService)
    service.provider=Provider()
    service.warnings=[]
    assert service.recognize(tmp_path/'image.png') is original
    assert len(service.warnings)==1
