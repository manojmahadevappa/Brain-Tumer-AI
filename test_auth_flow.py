"""
Test script to verify authentication flow
Run this to diagnose authentication issues
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("1. Testing Health Check")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
        return data.get('services', {}).get('firebase') == 'connected'
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_login(username, password):
    """Test login endpoint"""
    print("\n" + "="*60)
    print("2. Testing Login")
    print("="*60)
    try:
        response = requests.post(
            f"{BASE_URL}/api/login",
            json={"username": username, "password": password}
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Success: {data.get('success', False)}")
        if data.get('success'):
            print(f"Username: {data.get('username')}")
            print(f"Token (first 50 chars): {data.get('token', '')[:50]}...")
            print(f"Refresh Token present: {bool(data.get('refresh_token'))}")
            return data.get('token')
        else:
            print(f"Error: {data.get('error')}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_dashboard(token):
    """Test dashboard endpoint with token"""
    print("\n" + "="*60)
    print("3. Testing Dashboard API")
    print("="*60)
    if not token:
        print("❌ No token provided, skipping")
        return False
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/dashboard",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            print(f"✅ Success!")
            print(f"Total Analyses: {data.get('total_analyses', 0)}")
            print(f"Month Analyses: {data.get('month_analyses', 0)}")
            print(f"Total Messages: {data.get('total_messages', 0)}")
            return True
        else:
            print(f"❌ Failed: {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_debug_auth(token):
    """Test debug auth endpoint"""
    print("\n" + "="*60)
    print("4. Testing Debug Auth Endpoint")
    print("="*60)
    if not token:
        print("❌ No token provided, skipping")
        return
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/debug/auth",
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Debug Info: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("\n" + "="*60)
    print("🔐 Authentication Flow Test")
    print("="*60)
    
    # Test 1: Health check
    firebase_ok = test_health()
    if not firebase_ok:
        print("\n⚠️  Warning: Firebase may not be properly initialized")
    
    # Test 2: Login
    print("\n📝 Enter your credentials:")
    username = input("Username (or email): ").strip()
    password = input("Password: ").strip()
    
    if not username or not password:
        print("❌ Username and password required")
        return
    
    token = test_login(username, password)
    
    if not token:
        print("\n❌ Login failed. Check your credentials.")
        return
    
    # Test 3: Dashboard API
    dashboard_ok = test_dashboard(token)
    
    # Test 4: Debug auth
    test_debug_auth(token)
    
    # Summary
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    print(f"Firebase: {'✅' if firebase_ok else '⚠️'}")
    print(f"Login: {'✅' if token else '❌'}")
    print(f"Dashboard: {'✅' if dashboard_ok else '❌'}")
    
    if firebase_ok and token and dashboard_ok:
        print("\n✅ All tests passed! Authentication is working.")
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        if not firebase_ok:
            print("   - Firebase may need to be initialized properly")
        if not token:
            print("   - Login credentials may be incorrect")
        if token and not dashboard_ok:
            print("   - Token verification is failing")

if __name__ == "__main__":
    main()
