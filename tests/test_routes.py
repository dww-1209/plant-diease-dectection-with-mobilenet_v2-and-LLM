from io import BytesIO
from pathlib import Path

from plant_disease.config import Settings
from plant_disease.errors import LLMServiceError
from plant_disease.llm.mock_provider import MockProvider
from plant_disease.web.app import create_app


def test_get_treatment_advice_with_mock(client):
    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "番茄",
            "disease_name": "早疫病",
            "disease_degree": "一般",
            "health_status": "患病",
            "provider": "mock",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert "番茄" in data["advice"]


def test_get_treatment_advice_missing_fields(client):
    resp = client.post("/get_treatment_advice", json={"plant_class": "", "disease_name": ""})
    assert resp.status_code == 400
    assert resp.get_json()["success"] is False


def test_predict_no_file(client):
    resp = client.post("/predict")
    assert resp.status_code in (400, 503)


def test_predict_returns_503_when_init_failed(client):
    # 由于权重不存在，create_app 应把 InferenceModel 初始化失败保存下来；
    # /predict 在没有 image 字段时也走 400，但模型初始化失败的情况下应优先返回 503。
    resp = client.post(
        "/predict",
        data={"image": (BytesIO(b"x"), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 503
    assert "未初始化" in resp.get_json()["message"]


def test_get_treatment_advice_invalid_provider(client):
    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "a",
            "disease_name": "b",
            "disease_degree": "c",
            "health_status": "d",
            "provider": "no-such",
        },
    )
    assert resp.status_code == 400


class _FakeModel:
    def predict(self, _bytes):
        return {
            "class_id": 0,
            "probability": 0.9,
            "plant_class": "番茄",
            "health_status": "患病",
            "disease_name": "番茄早疫病",
            "disease_degree": "一般",
        }


def test_predict_success_path(settings):
    app = create_app(settings)
    app.config["INFERENCE_MODEL"] = _FakeModel()
    app.config["INIT_ERROR"] = None
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.post(
        "/predict",
        data={"image": (BytesIO(b"x"), "x.png")},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["success"] is True
    assert data["data"]["plant_class"] == "番茄"


class _BadProvider(MockProvider):
    def stream_treatment_advice(self, *args, **kwargs):
        raise LLMServiceError("upstream down")
        yield  # pragma: no cover - keep generator semantics


def test_get_treatment_advice_returns_502_on_service_error(settings):
    app = create_app(settings)
    app.config["LLM_PROVIDERS"] = {"mock": _BadProvider()}
    app.config["TESTING"] = True
    client = app.test_client()

    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "番茄",
            "disease_name": "早疫病",
            "disease_degree": "一般",
            "health_status": "患病",
            "provider": "mock",
        },
    )
    assert resp.status_code == 502
    assert "upstream down" in resp.get_json()["message"]


def test_get_treatment_advice_sse_stream(client):
    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "番茄",
            "disease_name": "早疫病",
            "disease_degree": "一般",
            "health_status": "患病",
            "provider": "mock",
        },
        headers={"Accept": "text/event-stream"},
    )
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    body = resp.get_data(as_text=True)
    assert "event: chunk" in body
    assert "event: done" in body
    assert "番茄" in body


def test_list_llm_providers(client):
    resp = client.get("/api/llm/providers")
    assert resp.status_code == 200
    payload = resp.get_json()["providers"]
    assert "openai" in payload and "deepseek" in payload
    assert "mock" in payload
    # 每家都有 default 和非空 models
    for spec in payload.values():
        assert spec["default"]
        assert isinstance(spec["models"], list) and spec["models"]


def test_advice_with_user_supplied_key_uses_one_off(settings, monkeypatch):
    """传 api_key 时走 build_one_off，且 service 不被进程缓存。"""
    app = create_app(settings)
    app.config["TESTING"] = True
    captured = {}

    def fake_one_off(provider, api_key, model):
        captured["provider"] = provider
        captured["api_key"] = api_key
        captured["model"] = model
        return MockProvider()

    monkeypatch.setattr("plant_disease.web.routes.build_one_off", fake_one_off)
    client = app.test_client()
    resp = client.post(
        "/get_treatment_advice",
        json={
            "plant_class": "番茄",
            "disease_name": "早疫病",
            "disease_degree": "一般",
            "health_status": "患病",
            "provider": "openai",
            "api_key": "sk-user-key",
            "model": "gpt-5.5",
        },
    )
    assert resp.status_code == 200
    assert captured == {"provider": "openai", "api_key": "sk-user-key", "model": "gpt-5.5"}
    # 不缓存：app.config["LLM_PROVIDERS"] 不应被一次性请求污染
    assert "openai" not in app.config.get("LLM_PROVIDERS", {})


def test_predict_rejects_oversized_upload():
    settings = Settings(
        weights_path=Path("missing.pth"),
        classes_txt=Path("missing.txt"),
        llm_provider="mock",
    )
    app = create_app(settings)
    app.config["INFERENCE_MODEL"] = _FakeModel()
    app.config["INIT_ERROR"] = None
    app.config["TESTING"] = True
    client = app.test_client()

    big = b"\x00" * (11 * 1024 * 1024)
    resp = client.post(
        "/predict",
        data={"image": (BytesIO(big), "big.bin")},
        content_type="multipart/form-data",
    )
    # Flask returns 413 when MAX_CONTENT_LENGTH is exceeded.
    assert resp.status_code == 413
