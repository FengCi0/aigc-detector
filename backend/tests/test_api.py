from backend.app import create_app


def test_health_endpoint():
    app = create_app()
    client = app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"


def test_detect_validation_error_for_short_text():
    app = create_app()
    client = app.test_client()
    response = client.post("/api/detect", json={"text": "short"})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "validation_error"


def test_detect_success():
    app = create_app()
    client = app.test_client()
    text = "这是一个用于接口测试的中文段落。" * 10
    response = client.post("/api/detect", json={"text": text, "include_details": True})
    assert response.status_code == 200
    payload = response.get_json()
    assert "aigc_score" in payload
    assert "confidence" in payload
    assert "features" in payload
    assert "model_mode" in payload
    assert "label" in payload
    assert "score_threshold" in payload
    assert "details" in payload
    assert "feature_contributions" in payload["details"]
    assert "calibration_enabled" in payload["details"]
    assert "calibration_metrics" in payload["details"]


def test_detect_include_details_string_false():
    app = create_app()
    client = app.test_client()
    text = "这是一个用于接口测试的中文段落。" * 10
    response = client.post("/api/detect", json={"text": text, "include_details": "false"})
    assert response.status_code == 200
    payload = response.get_json()
    assert "details" not in payload


def test_detect_score_threshold_consistent_with_label():
    app = create_app()
    client = app.test_client()
    text = "这是一个用于一致性测试的中文段落，长度足够，包含多个句子，方便验证分数阈值和最终标签是否一致。" * 5
    response = client.post("/api/detect", json={"text": text})
    assert response.status_code == 200
    payload = response.get_json()
    score = float(payload["aigc_score"])
    threshold = float(payload["score_threshold"])
    expected = "ai" if score >= threshold else "human"
    assert payload["label"] == expected


def test_detect_rejects_too_long_text(monkeypatch):
    monkeypatch.setenv("AIGC_MAX_TEXT_LENGTH", "500")
    app = create_app()
    client = app.test_client()
    text = "这是一段超长文本。" * 120
    response = client.post("/api/detect", json={"text": text})
    assert response.status_code == 400
    payload = response.get_json()
    assert payload["error"]["code"] == "validation_error"
