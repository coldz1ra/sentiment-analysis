from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json().get('status') == 'ok'


def test_predict_single():
    r = client.post('/predict', json={'text': 'Very dirty restroom'})
    assert r.status_code == 200
    data = r.json()
    assert 'label' in data and 'confidence' in data
    assert isinstance(data['confidence'], float)


def test_predict_batch():
    r = client.post('/predict-batch', json={'texts': ['Amazing product', 'Very dirty restroom']})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list) and len(data) == 2
    assert all(('label' in d and 'confidence' in d) for d in data)
