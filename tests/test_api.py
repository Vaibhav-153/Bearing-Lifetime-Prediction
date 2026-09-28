from fastapi.testclient import TestClient

from bearing_app.main import create_app


class FakePredictor:
    def __init__(self, result=42.0, error=None):
        self.result = result
        self.error = error

    def predict_rul(self, signals):
        if self.error is not None:
            raise self.error
        return self.result


def test_health_endpoint_reports_ready():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_prediction_endpoint_returns_prediction():
    payload = {"signals": [[0.1], [0.2]]}

    with TestClient(create_app(FakePredictor(result=17.5))) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predicted_rul": 17.5, "status": "success"}


def test_prediction_value_error_returns_400():
    with TestClient(create_app(FakePredictor(error=ValueError("bad input")))) as client:
        response = client.post("/predict", json={"signals": [[0.1]]})

    assert response.status_code == 400
    assert response.json()["detail"] == "bad input"


def test_prediction_schema_rejects_non_numeric_signal():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/predict", json={"signals": [["not-a-number"]]})

    assert response.status_code == 422
