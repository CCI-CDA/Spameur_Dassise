import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_check_spam(client):
    # Test avec un message de spam
    response = client.get("/check?message=SIX chances to win CASH!")
    assert response.status_code == 200
    assert response.json == {"resp": True}

    # Test avec un message normal
    response = client.get("/check?message=Hello, world, my name is Fred")
    assert response.status_code == 200
    assert response.json == {"resp": False}
