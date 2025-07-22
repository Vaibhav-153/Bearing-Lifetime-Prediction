# test_api.py
import requests
import json

# The URL where your API is running
API_URL = "http://127.0.0.1:8000/predict"

# The same sample data we used before
TEST_DATA = {
  "signals": [
    [0.1, -0.2, 0.1, 0.0, -0.1, 0.2, 0.3, -0.3, 0.1, 0.2],
    [0.2, -0.1, 0.2, 0.1, -0.2, 0.3, 0.2, -0.2, 0.0, 0.1],
    [0.1, 0.0, 0.1, 0.2, -0.1, 0.2, 0.4, -0.4, 0.2, 0.3],
    [0.3, -0.2, 0.2, 0.1, -0.1, 0.1, 0.3, -0.2, 0.1, 0.2],
    [0.2, -0.3, 0.1, 0.0, -0.2, 0.2, 0.2, -0.3, 0.1, 0.1],
    [0.1, -0.1, 0.2, 0.2, -0.1, 0.3, 0.4, -0.3, 0.1, 0.2],
    [0.4, -0.2, 0.1, 0.1, -0.1, 0.2, 0.3, -0.4, 0.0, 0.1],
    [0.2, 0.1, 0.1, 0.0, -0.2, 0.1, 0.2, -0.2, 0.1, 0.2],
    [0.3, -0.2, 0.2, 0.1, -0.1, 0.2, 0.5, -0.5, 0.1, 0.3],
    [0.5, -0.4, 0.3, 0.2, -0.3, 0.4, 0.6, -0.6, 0.2, 0.4],
    [0.4, -0.3, 0.4, 0.3, -0.2, 0.5, 0.7, -0.5, 0.3, 0.5],
    [0.6, -0.5, 0.5, 0.4, -0.4, 0.6, 0.8, -0.7, 0.4, 0.6],
    [0.7, -0.6, 0.6, 0.5, -0.5, 0.7, 0.9, -0.8, 0.5, 0.7],
    [0.8, -0.7, 0.7, 0.6, -0.6, 0.8, 1.0, -0.9, 0.6, 0.8],
    [0.9, -0.8, 0.8, 0.7, -0.7, 0.9, 1.1, -1.0, 0.7, 0.9]
  ]
}

print("Sending request to the Bearing RUL Prediction API...")

try:
    # Make the POST request
    response = requests.post(API_URL, json=TEST_DATA)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("\nRequest Successful!")
        print("API Response:")
        print(response.json())
    else:
        print(f"\nError: Received status code {response.status_code}")
        print("Response content:")
        print(response.text)

except requests.exceptions.ConnectionError as e:
    print(f"\nConnection Error: Could not connect to the API.")
    print("Please make sure the Uvicorn server is running.")