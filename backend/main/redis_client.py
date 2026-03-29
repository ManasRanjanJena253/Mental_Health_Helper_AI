"""
redis_client.py

Centralised async Redis client + lightweight helpers used across the app.

Responsibilities:
  - JWT blacklist  (for logout / token invalidation)
  - Rate-limit counters  (per user, per minute)
  - RunModel instance cache  (in-process TTLCache, Redis only stores the db_name string)

Why not pickle RunModel into Redis?
  RunModel holds live ChromaDB + LangChain objects that are not safely serialisable.
  Instead we keep the actual Python objects in a process-local TTLCache (cachetools)
  and use Redis only to persist the mapping user_name → chroma_db_name across restarts.
  On a cache miss we reconstruct cheaply from the db_name string.
"""

import os
from typing import Optional

import redis.asyncio as aioredis
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()


# Config
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# In-process model runner cache: max 128 users, 30-minute TTL
# Adjust MAX_RUNNERS down if memory is tight on your VPS.
MAX_RUNNERS: int = int(os.getenv("MODEL_RUNNER_CACHE_SIZE", "128"))
RUNNER_TTL_SECONDS: int = int(os.getenv("MODEL_RUNNER_TTL_SECONDS", "1800"))  # 30 min

# Redis key prefixes
_PREFIX_BLACKLIST = "jwt_blacklist:"
_PREFIX_RATE = "rate:"
_PREFIX_DB_NAME = "runner_db:"  # user_name → chroma db_name string

# Module-level singletons

# Lazily initialised async Redis pool — call `get_redis()` everywhere.
_redis_pool: Optional[aioredis.Redis] = None

# Process-local TTL cache for RunModel objects.
# Not thread-safe for writes by default; FastAPI runs on a single async loop
# so concurrent coroutine access is fine without a lock.
_runner_cache: TTLCache = TTLCache(maxsize=MAX_RUNNERS, ttl=RUNNER_TTL_SECONDS)



# Connection management
async def get_redis() -> aioredis.Redis:
    """Return the shared async Redis client, creating it on first call."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
    return _redis_pool


async def close_redis() -> None:
    """Gracefully close the Redis connection pool (call in app shutdown handler)."""
    global _redis_pool
    if _redis_pool:
        await _redis_pool.aclose()
        _redis_pool = None


# JWT blacklist helpers
# (used by /logout to invalidate tokens before they naturally expire)
async def blacklist_token(jti: str, ttl_seconds: int) -> None:
    """
    Add a token JTI (JWT ID) to the blacklist.
    TTL should match the token's remaining lifetime so Redis auto-cleans.

    Note: add 'jti' claim when creating tokens if you want per-token revocation.
    For simplicity we blacklist by user_name+iat concatenated as jti.
    """
    r = await get_redis()
    await r.setex(f"{_PREFIX_BLACKLIST}{jti}", ttl_seconds, "1")


async def is_token_blacklisted(jti: str) -> bool:
    r = await get_redis()
    return await r.exists(f"{_PREFIX_BLACKLIST}{jti}") == 1


# Rate limiting  (sliding window — simple counter approach)
async def check_rate_limit(user_name: str, max_requests: int = 30, window_seconds: int = 60) -> bool:
    """
    Returns True if the request is within limits, False if the user is throttled.
    Uses a simple fixed-window counter per minute.
    """
    r = await get_redis()
    key = f"{_PREFIX_RATE}{user_name}"
    count = await r.incr(key)
    if count == 1:
        # First request in this window — set expiry
        await r.expire(key, window_seconds)
    return count <= max_requests



# RunModel instance cache  (in-process TTLCache + Redis for db_name persistence)
def get_cached_runner(user_name: str):
    """Return cached RunModel instance or None on miss."""
    return _runner_cache.get(user_name)


def set_cached_runner(user_name: str, runner) -> None:
    """Store a RunModel instance in the in-process TTL cache."""
    _runner_cache[user_name] = runner


def evict_runner(user_name: str) -> None:
    """Manually evict a user's runner (e.g., on logout)."""
    _runner_cache.pop(user_name, None)


async def persist_runner_db_name(user_name: str, db_name: str) -> None:
    """
    Persist the chroma db_name to Redis so it survives process restarts.
    TTL = 7 days (longer than any refresh token).
    """
    r = await get_redis()
    await r.setex(f"{_PREFIX_DB_NAME}{user_name}", 60 * 60 * 24 * 7, db_name)


async def fetch_runner_db_name(user_name: str) -> Optional[str]:
    """Retrieve the chroma db_name from Redis. Returns None if not found."""
    r = await get_redis()
    return await r.get(f"{_PREFIX_DB_NAME}{user_name}")



# Health check
async def redis_ping() -> bool:
    """Returns True if Redis is reachable."""
    try:
        r = await get_redis()
        return await r.ping()
    except Exception:
        return False