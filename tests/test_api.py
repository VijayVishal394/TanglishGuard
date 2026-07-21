from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    response = client.post(
        "/predict",
        json={"text": "dei loose payale"}
    )

    assert response.status_code == 200
    assert "prediction" in response.json()