from datetime import datetime, timedelta, UTC
import jwt
from dotenv import load_dotenv
import os

load_dotenv()

# Get JWT secret from env
jwt_secret = os.getenv("JWT_SECRET_KEY")
if not jwt_secret:
    raise ValueError("JWT_SECRET_KEY not found in environment variables")

# Create payload
payload = {
    "type": "developer",
    "entity_id": "test_dev",
    "permissions": ["read", "write", "admin"],
    "exp": datetime.now(UTC) + timedelta(days=30)
}

# Generate token using secret from .env
token = jwt.encode(payload, jwt_secret, algorithm="HS256")

# Create URI
uri = f"databridge://test_dev:{token}@127.0.0.1:8000"
print(uri)
