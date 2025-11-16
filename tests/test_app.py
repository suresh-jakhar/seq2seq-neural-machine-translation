# API endpoint tests
import pytest
import json
from app import app


@pytest.fixture
def client():
    """Create a test client for the Flask application."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'healthy'


def test_home_page(client):
    """Test the home page serves HTML."""
    response = client.get('/')
    assert response.status_code == 200
    assert b'<!DOCTYPE html>' in response.data or b'<html>' in response.data


def test_translate_endpoint_success(client):
    """Test the translation endpoint with valid input."""
    response = client.post(
        '/translate',
        data=json.dumps({'text': 'Hello'}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'translated_text' in data
    assert 'original_text' in data
    assert data['original_text'] == 'Hello'
    assert 'success' in data


def test_translate_endpoint_empty_text(client):
    """Test the translation endpoint with empty text."""
    response = client.post(
        '/translate',
        data=json.dumps({'text': ''}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No text provided'


def test_translate_endpoint_missing_text(client):
    """Test the translation endpoint without text field."""
    response = client.post(
        '/translate',
        data=json.dumps({}),
        content_type='application/json'
    )
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
    assert data['error'] == 'No data provided'


def test_translate_endpoint_invalid_json(client):
    """Test the translation endpoint with invalid JSON."""
    response = client.post(
        '/translate',
        data='invalid json',
        content_type='application/json'
    )
    assert response.status_code in [400, 500]


def test_translate_endpoint_long_text(client):
    """Test the translation endpoint with long text."""
    long_text = 'Hello ' * 100  # Very long text
    response = client.post(
        '/translate',
        data=json.dumps({'text': long_text}),
        content_type='application/json'
    )
    # Should either succeed or return a proper error
    assert response.status_code in [200, 400]
    data = json.loads(response.data)
    assert 'success' in data


def test_translate_endpoint_special_characters(client):
    """Test the translation endpoint with special characters."""
    response = client.post(
        '/translate',
        data=json.dumps({'text': 'Hello! How are you?'}),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['success'] is True
    assert 'translated_text' in data
