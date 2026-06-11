"""
Quick test script — run locally or point BASE_URL at your Render deployment.

Usage:
    python test_api.py                            # local
    BASE_URL=https://your-app.onrender.com python test_api.py
"""

import os
import requests

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

def test_health():
    r = requests.get(f"{BASE_URL}/")
    print("Health:", r.json())

def test_info():
    r = requests.get(f"{BASE_URL}/info")
    print("Info:", r.json())

def test_predict_zeros():
    payload = {"features": [0.0] * 20}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Predict (zeros):", r.json())

def test_predict_random():
    import random
    payload = {"features": [random.gauss(0, 1) for _ in range(20)]}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Predict (random):", r.json())

def test_wrong_length():
    payload = {"features": [1.0] * 10}   # wrong length — should get 422
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Wrong length (expect 422):", r.status_code, r.json())

if __name__ == "__main__":
    test_health()
    test_info()
    test_predict_zeros()
    test_predict_random()
    test_wrong_length()
