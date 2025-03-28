import logging

# Initialize logger
logger = logging.getLogger(__name__)


async def check_and_increment_limits(auth, limit_type: str, value: int = 1) -> None:
    """
    Check if the user is within limits for an operation and increment usage.

    Args:
        auth: Authentication context with user_id
        limit_type: Type of limit to check (query, ingest, storage_file, storage_size, graph, cache)
        value: Value to check against limit (e.g., file size for storage_size)

    Raises:
        HTTPException: If the user exceeds limits
    """
    from fastapi import HTTPException
    from core.config import get_settings

    settings = get_settings()

    # Skip limit checking in self-hosted mode
    if settings.MODE == "self_hosted":
        return

    # Check if user_id is available
    if not auth.user_id:
        logger.warning("User ID not available in auth context, skipping limit check")
        return

    # Initialize user service
    from core.services.user_service import UserService

    user_service = UserService()
    await user_service.initialize()

    # Check if user is within limits
    within_limits = await user_service.check_limit(auth.user_id, limit_type, value)

    if not within_limits:
        # Get tier information for better error message
        user_data = await user_service.get_user_limits(auth.user_id)
        tier = user_data.get("tier", "unknown") if user_data else "unknown"

        # Map limit types to appropriate error messages
        limit_type_messages = {
            "query": f"Query limit exceeded for your {tier} tier. Please upgrade or try again later.",
            "ingest": f"Ingest limit exceeded for your {tier} tier. Please upgrade or try again later.",
            "storage_file": f"Storage file count limit exceeded for your {tier} tier. Please delete some files or upgrade.",
            "storage_size": f"Storage size limit exceeded for your {tier} tier. Please delete some files or upgrade.",
            "graph": f"Graph creation limit exceeded for your {tier} tier. Please upgrade to create more graphs.",
            "cache": f"Cache creation limit exceeded for your {tier} tier. Please upgrade to create more caches.",
            "cache_query": f"Cache query limit exceeded for your {tier} tier. Please upgrade or try again later.",
        }

        # Get message for the limit type or use default message
        detail = limit_type_messages.get(
            limit_type, f"Limit exceeded for your {tier} tier. Please upgrade or contact support."
        )

        # Raise the exception with appropriate message
        raise HTTPException(status_code=429, detail=detail)

    # Record usage asynchronously
    try:
        await user_service.record_usage(auth.user_id, limit_type, value)
    except Exception as e:
        # Just log if recording usage fails, don't fail the operation
        logger.error(f"Failed to record usage: {e}")
