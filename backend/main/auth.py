import os
from datetime import datetime, timedelta, UTC
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

load_dotenv()


# Config — set these in your .env
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION")  # openssl rand -hex 32
ALGORITHM: str = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# FastAPI will look for Bearer <token> in the Authorization header.
# tokenUrl must match your login endpoint path.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")



# Pydantic models
class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_name: Optional[str] = None



# Token creation
def _build_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(UTC) + expires_delta
    payload["iat"] = datetime.now(UTC)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_access_token(user_name: str) -> str:
    return _build_token(
        {"sub": user_name, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )


def create_refresh_token(user_name: str) -> str:
    return _build_token(
        {"sub": user_name, "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )


def create_token_pair(user_name: str) -> TokenPair:
    return TokenPair(
        access_token=create_access_token(user_name),
        refresh_token=create_refresh_token(user_name),
    )



# Token verification helpers
def _decode_token(token: str, expected_type: str) -> str:
    """
    Decode and validate a JWT.  Returns user_name on success.
    Raises HTTPException on any failure so callers stay clean.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise credentials_exception

    token_type: Optional[str] = payload.get("type")
    if token_type != expected_type:
        raise credentials_exception

    user_name: Optional[str] = payload.get("sub")
    if not user_name:
        raise credentials_exception

    return user_name


def verify_refresh_token(refresh_token: str) -> str:
    """Validate a refresh token and return the user_name."""
    return _decode_token(refresh_token, expected_type="refresh")


# FastAPI dependency — inject into any protected endpoint
async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """
    Dependency that extracts and validates the JWT from the Authorization header.
    Returns the authenticated user_name.

    Usage:
        @app.get("/protected")
        async def protected(user_name: str = Depends(get_current_user)):
            ...
    """
    return _decode_token(token, expected_type="access")



# Ensures the token owner matches a resource owner in the URL/body.
def assert_owns_resource(token_user: str, resource_user: str) -> None:
    """
    Raise 403 if the authenticated user is trying to access another user's data.
    Call this inside any endpoint that has a {user_name} path param.
    """
    if token_user != resource_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: you do not own this resource.",
        )