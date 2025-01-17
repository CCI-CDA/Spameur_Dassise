from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"message": "Test message"})
    assert response.status_code == 200
    assert "spam" in response.json()

def test_get_history():
    response = client.get("/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_login():
    response = client.post("/login", json={"email": "test@test.com", "password": "test123"})
    assert response.status_code == 200
    assert "access_token" in response.json()