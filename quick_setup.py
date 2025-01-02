import botocore
from dotenv import load_dotenv, find_dotenv
import os
import boto3
import logging
import tomli  # for reading toml files
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.operations import SearchIndexModel
import argparse

# Force reload of environment variables
load_dotenv(find_dotenv(), override=True)

# Set up argument parser
parser = argparse.ArgumentParser(description="Setup S3 bucket and MongoDB collections")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
parser.add_argument("--quiet", action="store_true", help="Only show warning and error logs")
args = parser.parse_args()

# Configure logging based on command line arguments
LOGGER = logging.getLogger(__name__)
match (args.debug, args.quiet):
    case (True, _):
        LOGGER.setLevel(logging.DEBUG)
    case (_, True):
        LOGGER.setLevel(logging.WARNING)
    case _:
        LOGGER.setLevel(logging.INFO)

# Add console handler with formatting
console_handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
LOGGER.addHandler(console_handler)

# Load configuration from config.toml
config_path = Path("config.toml")
with open(config_path, "rb") as f:
    CONFIG = tomli.load(f)
    LOGGER.info("Loaded configuration from config.toml")

# Extract configuration values
STORAGE_PROVIDER = CONFIG["service"]["components"]["storage"]
DATABASE_PROVIDER = CONFIG["service"]["components"]["database"]
DATABASE_NAME = CONFIG["database"][DATABASE_PROVIDER]["database_name"]

# MongoDB specific config
DOCUMENTS_COLLECTION = CONFIG["database"]["mongodb"]["documents_collection"]
CHUNKS_COLLECTION = CONFIG["database"]["mongodb"]["chunks_collection"]
VECTOR_DIMENSIONS = CONFIG["vector_store"]["mongodb"]["dimensions"]
VECTOR_INDEX_NAME = CONFIG["vector_store"]["mongodb"]["index_name"]
SIMILARITY_METRIC = CONFIG["vector_store"]["mongodb"]["similarity_metric"]

# PostgreSQL specific config
DOCUMENTS_TABLE = CONFIG["database"]["postgres"]["documents_table"]
CHUNKS_TABLE = CONFIG["database"]["postgres"]["chunks_table"]

# Extract storage-specific configuration
DEFAULT_REGION = CONFIG["storage"]["aws"]["region"] if STORAGE_PROVIDER == "aws-s3" else None
DEFAULT_BUCKET_NAME = (
    CONFIG["storage"]["aws"]["bucket_name"] if STORAGE_PROVIDER == "aws-s3" else None
)


def create_s3_bucket(bucket_name, region=DEFAULT_REGION):
    """Set up S3 bucket."""
    # Clear any existing AWS credentials from environment
    boto3.Session().resource("s3").meta.client.close()

    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    region = os.getenv("AWS_REGION") if os.getenv("AWS_REGION") else region

    if not aws_access_key or not aws_secret_key:
        LOGGER.error("AWS credentials not found in environment variables.")
        return

    LOGGER.debug("Successfully retrieved AWS credentials and region.")
    # Create new session with explicit credentials
    session = boto3.Session(
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region,
    )

    s3_client = session.client("s3")
    LOGGER.debug("Successfully created S3 client.")

    if bucket_exists(s3_client, bucket_name):
        LOGGER.info(f"Bucket with name {bucket_name} already exists")
        return

    if region == "us-east-1":
        s3_client.create_bucket(Bucket=bucket_name)
    else:
        s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": region}
        )

    LOGGER.debug(f"Bucket {bucket_name} created successfully in {region} region.")


def bucket_exists(s3_client, bucket_name):
    """Check if an S3 bucket exists."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except botocore.exceptions.ClientError as e:
        error_code = int(e.response["Error"]["Code"])
        if error_code == 404:
            return False
        raise


def setup_mongodb():
    """
    Set up MongoDB database, documents collection, and vector index on documents_chunk collection.
    """
    # Load MongoDB URI from .env file
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise ValueError("MONGODB_URI not found in .env file.")

    try:
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        client.admin.command("ping")  # Check connection
        LOGGER.info("Connected to MongoDB successfully.")

        # Create or access the database
        db = client[DATABASE_NAME]
        LOGGER.info(f"Database '{DATABASE_NAME}' ready.")

        # Create 'documents' collection
        if DOCUMENTS_COLLECTION not in db.list_collection_names():
            db.create_collection(DOCUMENTS_COLLECTION)
            LOGGER.info(f"Collection '{DOCUMENTS_COLLECTION}' created.")
        else:
            LOGGER.info(f"Collection '{DOCUMENTS_COLLECTION}' already exists.")

        # Create 'documents_chunk' collection with vector index
        if CHUNKS_COLLECTION not in db.list_collection_names():
            db.create_collection(CHUNKS_COLLECTION)
            LOGGER.info(f"Collection '{CHUNKS_COLLECTION}' created.")
        else:
            LOGGER.info(f"Collection '{CHUNKS_COLLECTION}' already exists.")

        vector_index_definition = {
            "fields": [
                {
                    "numDimensions": VECTOR_DIMENSIONS,
                    "path": "embedding",
                    "similarity": SIMILARITY_METRIC,
                    "type": "vector",
                },
                {"path": "document_id", "type": "filter"},
            ]
        }
        vector_index = SearchIndexModel(
            name=VECTOR_INDEX_NAME,
            definition=vector_index_definition,
            type="vectorSearch",
        )
        db[CHUNKS_COLLECTION].create_search_index(model=vector_index)
        LOGGER.info("Vector index 'vector_index' created on 'documents_chunk' collection.")

    except ConnectionFailure:
        LOGGER.error("Failed to connect to MongoDB. Check your MongoDB URI and network connection.")
    except OperationFailure as e:
        LOGGER.error(f"MongoDB operation failed: {e}")
    except Exception as e:
        LOGGER.error(f"Unexpected error: {e}")
    finally:
        client.close()
        LOGGER.info("MongoDB connection closed.")


def setup_postgres():
    """
    Set up PostgreSQL database and tables with proper indexes.
    """
    import asyncio
    from sqlalchemy.ext.asyncio import create_async_engine

    # Load PostgreSQL URI from .env file
    postgres_uri = os.getenv("POSTGRES_URI")
    if not postgres_uri:
        raise ValueError("POSTGRES_URI not found in .env file.")

    async def _setup_postgres():
        try:
            # Create async engine
            engine = create_async_engine(postgres_uri)

            async with engine.begin() as conn:
                # Import and create all tables
                from core.database.postgres_database import Base

                await conn.run_sync(Base.metadata.create_all)
                LOGGER.info("Created all PostgreSQL tables and indexes.")

            await engine.dispose()
            LOGGER.info("PostgreSQL setup completed successfully.")

        except Exception as e:
            LOGGER.error(f"Failed to setup PostgreSQL: {e}")
            raise

    asyncio.run(_setup_postgres())


def setup():
    # Setup S3 if configured
    if STORAGE_PROVIDER == "aws-s3":
        LOGGER.info("Setting up S3 bucket...")
        create_s3_bucket(DEFAULT_BUCKET_NAME, DEFAULT_REGION)
        LOGGER.info("S3 bucket setup completed.")

    # Setup database based on provider
    match DATABASE_PROVIDER:
        case "mongodb":
            LOGGER.info("Setting up MongoDB...")
            setup_mongodb()
            LOGGER.info("MongoDB setup completed.")
        case "postgres":
            LOGGER.info("Setting up PostgreSQL...")
            setup_postgres()
            LOGGER.info("PostgreSQL setup completed.")
        case _:
            LOGGER.error(f"Unsupported database provider: {DATABASE_PROVIDER}")
            raise ValueError(f"Unsupported database provider: {DATABASE_PROVIDER}")

    LOGGER.info("Setup completed successfully. Feel free to start the server now!")


if __name__ == "__main__":
    setup()
