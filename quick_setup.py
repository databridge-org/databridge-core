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
import platform
import subprocess

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

# Load configuration from databridge.toml
config_path = Path("databridge.toml")
with open(config_path, "rb") as f:
    CONFIG = tomli.load(f)
    LOGGER.info("Loaded configuration from databridge.toml")

# Extract configuration values
STORAGE_PROVIDER = CONFIG["storage"]["provider"]
DATABASE_PROVIDER = CONFIG["database"]["provider"]

# MongoDB specific config
if "mongodb" in CONFIG["database"]:
    DATABASE_NAME = CONFIG["database"]["mongodb"]["database_name"]
    DOCUMENTS_COLLECTION = "documents"
    CHUNKS_COLLECTION = "document_chunks"
    if "mongodb" in CONFIG["vector_store"]:
        VECTOR_DIMENSIONS = CONFIG["embedding"]["dimensions"]
        VECTOR_INDEX_NAME = "vector_index"
        SIMILARITY_METRIC = CONFIG["embedding"]["similarity_metric"]

# Extract storage-specific configuration
if STORAGE_PROVIDER == "aws-s3":
    DEFAULT_REGION = CONFIG["storage"]["region"]
    DEFAULT_BUCKET_NAME = CONFIG["storage"]["bucket_name"]
else:
    DEFAULT_REGION = None
    DEFAULT_BUCKET_NAME = None


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
    from sqlalchemy import text

    # Load PostgreSQL URI from .env file
    postgres_uri = os.getenv("POSTGRES_URI")
    if not postgres_uri:
        raise ValueError("POSTGRES_URI not found in .env file.")

    # Check if pgvector is installed when on macOS
    if platform.system() == "Darwin":
        try:
            # Check if postgresql is installed via homebrew
            result = subprocess.run(
                ["brew", "list", "postgresql@14"], capture_output=True, text=True
            )
            if result.returncode != 0:
                LOGGER.error(
                    "PostgreSQL not found. Please install it with: brew install postgresql@14"
                )
                raise RuntimeError("PostgreSQL not installed")

            # Check if pgvector is installed
            result = subprocess.run(["brew", "list", "pgvector"], capture_output=True, text=True)
            if result.returncode != 0:
                LOGGER.error(
                    "\nError: pgvector extension not found. Please install it with:\n"
                    "brew install pgvector\n"
                    "brew services stop postgresql@14\n"
                    "brew services start postgresql@14\n"
                )
                raise RuntimeError("pgvector not installed")
        except FileNotFoundError:
            LOGGER.error("Homebrew not found. Please install it from https://brew.sh")
            raise

    async def _setup_postgres():
        try:
            # Create async engine
            engine = create_async_engine(postgres_uri)

            async with engine.begin() as conn:
                try:
                    # Enable pgvector extension
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    LOGGER.info("Enabled pgvector extension")
                except Exception as e:
                    if "could not open extension control file" in str(e):
                        LOGGER.error(
                            "\nError: pgvector extension not found. Please install it:\n"
                            "- On macOS: brew install pgvector\n"
                            "- On Ubuntu: sudo apt install postgresql-14-pgvector\n"
                            "- On other systems: check https://github.com/pgvector/pgvector#installation\n"
                        )
                    raise

                # Import and create all tables
                from core.database.postgres_database import Base

                # Create regular tables first
                await conn.run_sync(Base.metadata.create_all)
                LOGGER.info("Created base PostgreSQL tables")

                # Create caches table
                create_caches_table = """
                CREATE TABLE IF NOT EXISTS caches (
                    name TEXT PRIMARY KEY,
                    metadata JSON NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """
                await conn.execute(text(create_caches_table))
                LOGGER.info("Created caches table")

                # Get vector dimensions from config
                dimensions = CONFIG["embedding"]["dimensions"]

                # Drop existing vector index if it exists
                drop_index_sql = """
                DROP INDEX IF EXISTS vector_idx;
                """
                await conn.execute(text(drop_index_sql))

                # Drop existing vector embeddings table if it exists
                drop_table_sql = """
                DROP TABLE IF EXISTS vector_embeddings;
                """
                await conn.execute(text(drop_table_sql))

                # Create vector embeddings table with proper vector column
                create_table_sql = f"""
                CREATE TABLE vector_embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255),
                    chunk_number INTEGER,
                    content TEXT,
                    chunk_metadata TEXT,
                    embedding vector({dimensions}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                """
                await conn.execute(text(create_table_sql))
                LOGGER.info("Created vector_embeddings table with vector column")

                # Create the vector index
                index_sql = """
                CREATE INDEX vector_idx
                ON vector_embeddings USING ivfflat (embedding vector_l2_ops)
                WITH (lists = 100);
                """
                await conn.execute(text(index_sql))
                LOGGER.info("Created IVFFlat index on vector_embeddings")

            await engine.dispose()
            LOGGER.info("PostgreSQL setup completed successfully")

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
