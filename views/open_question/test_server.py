#!/usr/bin/env python3
"""
Test script for the Open Question Search Server
"""

import requests
import json
import time

def test_server():
    """Test the server endpoints"""
    base_url = "http://localhost:8000"
    
    print("Testing server endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✓ Health endpoint working")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Make sure it's running on port 8000")
        return False
    
    # Test search endpoint
    try:
        test_query = "test query"
        response = requests.post(
            f"{base_url}/search",
            json={"query": test_query},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ Search endpoint working")
            print(f"  Query: '{test_query}'")
            print(f"  Response: {data.get('message', 'No message')}")
            print(f"  Scores generated: {len(data.get('scores', {}))}")
        else:
            print(f"✗ Search endpoint failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Search test failed: {e}")
        return False
    
    # Test scores endpoint
    try:
        response = requests.get(f"{base_url}/scores")
        if response.status_code == 200:
            data = response.json()
            print("✓ Scores endpoint working")
            print(f"  Scores available: {len(data.get('scores', {}))}")
        else:
            print(f"✗ Scores endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Scores test failed: {e}")
        return False
    
    print("\n✓ All tests passed! Server is working correctly.")
    return True

if __name__ == "__main__":
    print("=== Server Test ===")
    test_server() 