from pathlib import Path
import sys
import pytest
from typing import Generator
import os
from dotenv import load_dotenv

root_dir = Path(__file__).parent.parent.parent
sdk_path = str(root_dir / "sdks" / "python")
core_path = str(root_dir)

sys.path.extend([sdk_path, core_path])

from core.config import get_settings
from databridge import DataBridge
# Load test environment variables
load_dotenv(".env.test")


@pytest.fixture(scope="session")
def settings():
    """Get test settings"""
    return get_settings()


@pytest.fixture
async def db() -> Generator[DataBridge, None, None]:
    """DataBridge client fixture"""
    uri = os.getenv("DATABRIDGE_TEST_URI")
    if not uri:
        raise ValueError("DATABRIDGE_TEST_URI not set")
        
    client = DataBridge(uri)
    try:
        yield client
    finally:
        await client.close()
