from app.services.reconstruction.router import ReconstructionRouter


def test_router_keeps_text_editable_and_rejects_low_confidence_shape():
    router = ReconstructionRouter()
    assert router.route({"type": "text", "confidence": 0.2}) == "editable_text"
    assert router.route({"type": "rectangle", "confidence": 0.4}) == "local_image"
    assert router.route({"type": "rectangle", "confidence": 0.9}) == "native_shape"


def test_router_honors_background_and_explicit_strategy():
    router = ReconstructionRouter()
    assert router.route({"type": "background"}) == "background_image"
    assert router.route({"type": "image", "metadata": {"reconstructionStrategy": "transparent_image"}}) == "transparent_image"
