"""Smoke test for the FastAPI endpoints."""
import sys
import time
import requests

API_URL = 'http://localhost:8000'

def test_health():
    print('Testing /health...')
    r = requests.get(f'{API_URL}/health')
    assert r.status_code == 200, 'Health check failed'
    print('Health OK:', r.json())

