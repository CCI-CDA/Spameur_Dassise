from fastapi.testclient import TestClient
import pytest
from app import app, create_access_token

client = TestClient(app)

# Créer un token de test
test_token = create_access_token({"sub": "test@example.com"})
headers = {"Authorization": f"Bearer {test_token}"}

def test_login():
    response = client.post("/login", json={
        "username": "dassise",
        "password": "1234"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_predict():
    response = client.post(
        "/predict",
        headers=headers,
        json={"message": "Win a free iPhone now!"}
    )
    assert response.status_code == 200
    assert "spam" in response.json()
    assert "probability" in response.json()
    assert "analysis" in response.json()

def test_batch_predict():
    # Créer un fichier CSV de test
    csv_content = b"message\nWin a free iPhone!\nHello, how are you?"
    response = client.post(
        "/batch_predict",
        headers=headers,
        files={"file": ("test.csv", csv_content, "text/csv")}
    )
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_history():
    response = client.get("/history", headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_quota():
    response = client.get("/quota", headers=headers)
    assert response.status_code == 200
    assert "remaining" in response.json() 