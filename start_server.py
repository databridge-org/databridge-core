import uvicorn
import os
from dotenv import load_dotenv


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Verify required environment variables
    required_vars = [
        "MONGODB_URI",
        "OPENAI_API_KEY",
        "UNSTRUCTURED_API_KEY",
        "JWT_SECRET_KEY"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    # Start server
    uvicorn.run(
        "core.api:app",
        host="0.0.0.0",  # Listen on all available interfaces
        port=8000,
        reload=True,  # Enable auto-reload during development
        loop="asyncio"
    )

if __name__ == "__main__":
    main()
