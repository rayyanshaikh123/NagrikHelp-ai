from fastapi.testclient import TestClient

from local_vision_server import app, MODEL_NAME


def test_root_returns_ok_and_model():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("ok") is True
    assert body.get("model") == MODEL_NAME
