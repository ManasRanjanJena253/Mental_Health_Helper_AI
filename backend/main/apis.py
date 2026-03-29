"""
apis.py
~~~~~~~
FastAPI application — refactored with:
  - JWT authentication on all protected endpoints
  - Redis-backed RunModel cache (in-process TTLCache, Redis for persistence)
  - SSE streaming on /chat  (tokens arrive at the frontend as they generate)
  - All sync blocking calls moved to run_in_executor (event loop never blocked)
  - Voice temp files use uuid names to prevent collision under concurrency
  - Rate limiting on /chat and /voice endpoints
"""

import asyncio
import json
import os
import shutil
import uuid
from datetime import datetime, UTC
from functools import partial
from typing import Optional, AsyncIterator

import uvicorn
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorClient
from passlib.hash import bcrypt
from pymongo.server_api import ServerApi
from langchain_google_genai import ChatGoogleGenerativeAI

# Local modules
from auth import (
    TokenPair,
    assert_owns_resource,
    create_token_pair,
    get_current_user,
    verify_refresh_token,
)
from redis_client import (
    check_rate_limit,
    close_redis,
    evict_runner,
    fetch_runner_db_name,
    get_cached_runner,
    persist_runner_db_name,
    redis_ping,
    set_cached_runner,
)
from chat_runner import RunModel

load_dotenv()

# App + DB setup

uri = os.getenv("MONGO_URI")
app = FastAPI(title="MindHaven API", version="2.0.0")

mongo_client = AsyncIOMotorClient(uri, server_api=ServerApi("1"))
db = mongo_client["Mental_Health_AI_User_Database"]
collection = db["user_data"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifecycle events
@app.on_event("startup")
async def startup():
    ok = await redis_ping()
    if not ok:
        raise RuntimeError("Redis is unreachable — check REDIS_URL in .env")


@app.on_event("shutdown")
async def shutdown():
    await close_redis()
    mongo_client.close()


# Internal helpers
def _get_or_create_runner(user_name: str) -> RunModel:
    """
    Return a cached RunModel for user_name, creating it on a cache miss.
    This is synchronous and designed to be called via run_in_executor.
    """
    runner = get_cached_runner(user_name)
    if runner is None:
        db_name = f"VectorDB_{user_name}"
        runner = RunModel(db_name=db_name)
        set_cached_runner(user_name, runner)
    return runner


async def _get_runner(user_name: str) -> RunModel:
    """Async wrapper: checks in-process cache, then Redis, then constructs."""
    runner = get_cached_runner(user_name)
    if runner is not None:
        return runner

    # Try to recover db_name from Redis (survives process restarts)
    db_name = await fetch_runner_db_name(user_name) or f"VectorDB_{user_name}"

    loop = asyncio.get_event_loop()
    runner = await loop.run_in_executor(None, partial(RunModel, db_name=db_name))
    set_cached_runner(user_name, runner)
    # Persist db_name so the next cold start can find it
    await persist_runner_db_name(user_name, db_name)
    return runner


def _create_chat_name_sync(user_prompt: str) -> str:
    """Sync LLM call to generate a chat title — run in executor."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )
    response = llm.invoke(
        f"This is the prompt given by the user: {user_prompt}\n"
        "Provide a suitable short title for this chat. "
        "Only the title — no greetings, no extra text."
    )
    return response.content


def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit."
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter."
    return True, ""


async def _get_user_or_404(user_name: str) -> dict:
    doc = await collection.find_one({"user_name": user_name})
    if not doc:
        raise HTTPException(status_code=404, detail="User not found.")
    return doc


async def _get_cipher(user_name: str) -> tuple[dict, Fernet]:
    doc = await _get_user_or_404(user_name)
    key = doc.get("encryption_key")
    if not key:
        raise HTTPException(status_code=500, detail="Encryption key missing.")
    return doc, Fernet(key)


# Public endpoints  (no JWT required)
@app.get("/health")
async def health():
    redis_ok = await redis_ping()
    return {"status": "ok", "redis": redis_ok}


@app.post("/sign_in")
async def sign_in(user_name: str = Form(...), password: str = Form(...)):
    """Register a new user."""
    existing = await collection.find_one({"user_name": user_name})
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken.")

    is_valid, err = validate_password(password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err)

    loop = asyncio.get_event_loop()
    hashed = await loop.run_in_executor(None, partial(bcrypt.hash, password))
    key = Fernet.generate_key()

    await collection.insert_one({
        "user_name": user_name,
        "password": hashed,
        "encryption_key": key,
        "session_ids": [],
        "chat_names": [],
        "chats": [],
        "created_at": datetime.now(UTC),
    })

    return {"detail": "Account created. Please log in."}


@app.post("/login", response_model=TokenPair)
async def login(user_name: str = Form(...), password: str = Form(...)):
    """Authenticate and return a JWT access + refresh token pair."""
    doc = await collection.find_one({"user_name": user_name}, {"password": 1})
    if not doc:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    loop = asyncio.get_event_loop()
    match = await loop.run_in_executor(
        None, partial(bcrypt.verify, password, doc["password"])
    )
    if not match:
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    return create_token_pair(user_name)


@app.post("/refresh", response_model=TokenPair)
async def refresh_tokens(refresh_token: str = Form(...)):
    """Exchange a valid refresh token for a new access + refresh token pair."""
    user_name = verify_refresh_token(refresh_token)
    return create_token_pair(user_name)


# Protected endpoints  (JWT required via Depends(get_current_user))
@app.post("/logout")
async def logout(current_user: str = Depends(get_current_user)):
    """Evict the user's model runner from the in-process cache on logout."""
    evict_runner(current_user)
    return {"detail": "Logged out."}


@app.get("/{user_name}/all_sessions")
async def list_sessions(
    user_name: str,
    current_user: str = Depends(get_current_user),
):
    assert_owns_resource(current_user, user_name)

    doc, cipher = await _get_cipher(user_name)
    chat_names = []
    for enc_name in doc.get("chat_names", []):
        try:
            chat_names.append(cipher.decrypt(enc_name).decode("utf-8"))
        except Exception:
            chat_names.append("Untitled Chat")

    return {
        "session_ids": doc.get("session_ids", []),
        "chat_names": chat_names,
    }


@app.post("/{user_name}/new_session")
async def new_session(
    user_name: str,
    current_user: str = Depends(get_current_user),
):
    assert_owns_resource(current_user, user_name)
    await _get_user_or_404(user_name)

    for _ in range(10):
        sid = str(uuid.uuid4())
        clash = await collection.find_one({"user_name": user_name, "session_ids": sid})
        if not clash:
            return {"session_id": sid}

    raise HTTPException(status_code=500, detail="Could not generate unique session ID.")


@app.post("/{user_name}/{session_id}/chat")
async def chat(
    user_name: str,
    session_id: str,
    user_prompt: str = Form(...),
    current_user: str = Depends(get_current_user),
):
    """
    SSE streaming chat endpoint.
    Tokens are yielded as `data: <token>\\n\\n` events.
    The final event is `data: [DONE]\\n\\n`.
    The frontend should use EventSource or fetch with a ReadableStream.
    """
    assert_owns_resource(current_user, user_name)

    # Rate limit: 30 messages per minute per user
    allowed = await check_rate_limit(user_name, max_requests=30, window_seconds=60)
    if not allowed:
        raise HTTPException(status_code=429, detail="Too many requests. Slow down a bit.")

    doc, cipher = await _get_cipher(user_name)
    session_ids: list = doc.get("session_ids", [])
    is_new_session = session_id not in session_ids

    loop = asyncio.get_event_loop()

    if is_new_session:
        # Generate chat name without blocking the event loop
        chat_name = await loop.run_in_executor(
            None, partial(_create_chat_name_sync, user_prompt)
        )
        enc_name = cipher.encrypt(chat_name.encode("utf-8"))
        await collection.update_one(
            {"user_name": user_name},
            {"$push": {"session_ids": session_id, "chat_names": enc_name, "chats": []}},
        )
    else:
        # Ensure chats array is correctly sized (guard against schema drift)
        session_idx = session_ids.index(session_id)
        chats = doc.get("chats", [])
        while len(chats) <= session_idx:
            await collection.update_one(
                {"user_name": user_name}, {"$push": {"chats": []}}
            )
            chats.append([])

    runner = await _get_runner(user_name)

    async def event_generator() -> AsyncIterator[str]:
        full_parts: list[str] = []
        try:
            async for token in runner.astream(user_prompt, session_id, user_name):
                full_parts.append(token)
                # Escape newlines inside the SSE data field
                safe_token = token.replace("\n", "\\n")
                yield f"data: {safe_token}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"
            return

        # Persist message to MongoDB after streaming completes
        full_response = "".join(full_parts)
        if full_response:
            enc_user = cipher.encrypt(user_prompt.encode("utf-8"))
            enc_ai = cipher.encrypt(full_response.encode("utf-8"))
            msg_obj = {
                "user_message": enc_user,
                "ai_message": enc_ai,
                "timestamp": datetime.now(UTC),
            }
            # Re-fetch to get accurate session index after any concurrent updates
            updated = await collection.find_one({"user_name": user_name})
            curr_ids = updated.get("session_ids", [])
            if session_id in curr_ids:
                idx = curr_ids.index(session_id)
                await collection.update_one(
                    {"user_name": user_name},
                    {"$push": {f"chats.{idx}": msg_obj}},
                )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Disable Nginx buffering if behind a proxy
        },
    )


@app.delete("/delete_chat/{user_name}/{session_id}")
async def delete_chat(
    user_name: str,
    session_id: str,
    current_user: str = Depends(get_current_user),
):
    assert_owns_resource(current_user, user_name)
    doc = await _get_user_or_404(user_name)

    session_ids: list = doc.get("session_ids", [])
    if session_id not in session_ids:
        raise HTTPException(status_code=404, detail="Session not found.")

    idx = session_ids.index(session_id)
    chat_names = doc.get("chat_names", [])
    chats = doc.get("chats", [])

    new_session_ids = session_ids[:idx] + session_ids[idx + 1:]
    new_chat_names = chat_names[:idx] + chat_names[idx + 1:] if idx < len(chat_names) else chat_names
    new_chats = chats[:idx] + chats[idx + 1:] if idx < len(chats) else chats

    await collection.update_one(
        {"user_name": user_name},
        {"$set": {"session_ids": new_session_ids, "chat_names": new_chat_names, "chats": new_chats}},
    )
    return {"detail": "Chat deleted."}


@app.get("/{user_name}/{session_id}/messages")
async def get_messages(
    user_name: str,
    session_id: str,
    limit: Optional[int] = 50,
    current_user: str = Depends(get_current_user),
):
    assert_owns_resource(current_user, user_name)

    doc, cipher = await _get_cipher(user_name)
    session_ids: list = doc.get("session_ids", [])
    if session_id not in session_ids:
        raise HTTPException(status_code=404, detail="Session not found.")

    idx = session_ids.index(session_id)
    chats = doc.get("chats", [])
    if idx >= len(chats):
        return {"messages": []}

    messages = []
    for msg in (chats[idx] or [])[-limit:]:
        try:
            messages.append({
                "user_message": cipher.decrypt(msg["user_message"]).decode("utf-8"),
                "ai_message": cipher.decrypt(msg["ai_message"]).decode("utf-8"),
                "timestamp": msg.get("timestamp", ""),
            })
        except Exception:
            continue

    return {"messages": messages}



# Voice endpoint
@app.post("/{user_name}/{session_id}/chat/voice")
async def voice_chat(
    user_name: str,
    session_id: str,
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user),
):
    assert_owns_resource(current_user, user_name)

    allowed = await check_rate_limit(user_name, max_requests=10, window_seconds=60)
    if not allowed:
        raise HTTPException(status_code=429, detail="Voice rate limit exceeded.")

    # Use uuid-based temp path to prevent collisions under concurrent requests
    ext = os.path.splitext(file.filename or "audio")[1] or ".tmp"
    temp_path = f"/tmp/mh_voice_{uuid.uuid4()}{ext}"

    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        from voice_bridge import MindHavenVoice
        voice = MindHavenVoice()

        loop = asyncio.get_event_loop()
        user_prompt = await loop.run_in_executor(
            None, partial(voice.speech_to_text, audio_file=temp_path)
        )

        doc, cipher = await _get_cipher(user_name)
        session_ids: list = doc.get("session_ids", [])
        is_new_session = session_id not in session_ids

        if is_new_session:
            chat_name = await loop.run_in_executor(
                None, partial(_create_chat_name_sync, user_prompt)
            )
            enc_name = cipher.encrypt(chat_name.encode("utf-8"))
            await collection.update_one(
                {"user_name": user_name},
                {"$push": {"session_ids": session_id, "chat_names": enc_name, "chats": []}},
            )

        runner = await _get_runner(user_name)
        response = await runner.arun(user_prompt, session_id, user_name)

        # Stream TTS audio back
        audio_stream = voice.gtts_stream(text=response)

        # Persist to MongoDB (fire-and-forget)
        async def _persist():
            enc_user = cipher.encrypt(user_prompt.encode("utf-8"))
            enc_ai = cipher.encrypt(response.encode("utf-8"))
            updated = await collection.find_one({"user_name": user_name})
            curr_ids = updated.get("session_ids", [])
            if session_id in curr_ids:
                idx = curr_ids.index(session_id)
                await collection.update_one(
                    {"user_name": user_name},
                    {"$push": {f"chats.{idx}": {
                        "user_message": enc_user,
                        "ai_message": enc_ai,
                        "timestamp": datetime.now(UTC),
                    }}},
                )

        asyncio.create_task(_persist())

        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=reply.mp3",
                "Cache-Control": "no-cache",
            },
        )

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8001, reload=False)