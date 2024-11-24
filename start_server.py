import uvicorn
from dotenv import load_dotenv
from core.config import get_settings


def main():
    # Load environment variables from .env file
    load_dotenv()

    # Load settings (this will validate all required env vars)
    settings = get_settings()

    # Start server
    uvicorn.run(
        "core.api:app",
        host=settings.HOST,
        port=settings.PORT,
        # reload=settings.RELOAD
    )


if __name__ == "__main__":
    main()
