import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import uuid
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pymongo.server_api import ServerApi
from backend.main.chat_runner import RunModel
from passlib.hash import bcrypt
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, List, Dict

########################################################################################################################################################################
# How bcrypt Works (Step by Step)
# User chooses a password, e.g. "mypassword123".
# Generate a salt (random string, usually 128 bits).
# Example salt: "a8f5f167f44f4964e6c998dee827110c".
# This ensures that even if two users have the same password, their hashes are different.
# Password + Salt → Key Expansion & Hashing
# bcrypt uses the Blowfish cipher internally.
# It runs the password through multiple expensive rounds of encryption + mixing with the salt.
# Cost Factor (Work Factor)
# bcrypt has a tunable parameter called cost (also called "rounds").
# Example: cost = 12 → means the hashing function runs 2^12 = 4096 iterations internally.
# Higher cost = more secure but slower.
# Final Hash Stored
# bcrypt outputs a hash like this (60 characters):
# $2b$12$eImiTXuWVxfM37uY4JANjQ==.r9pT7hJ7W0mTgk0U1fM/6Jb5PScq
# Format breakdown:
# $2b$ → bcrypt version
# 12 → cost factor (work factor)
# Next 22 chars → salt
# Last part → actual hashed password
#########################################################################################################################################################################

# Using bcrypt for passwords because they can't be decrypted.
# Using encryption for storing chat_names.

load_dotenv()

uri = os.getenv("MONGO_URI")
app = FastAPI()

client = AsyncIOMotorClient(uri, server_api = ServerApi("1"))
db_name = "Mental_Health_AI_User_Database"
db = client[db_name]
collection = db["user_data"]

# Initialize RunModel once to avoid repeated instantiation
model_runner_cache = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # list of origins; use ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],  # allow all HTTP methods
    allow_headers=["*"],  # allow all headers
)


def create_chat_name(user_prompt: str) -> str:
    """
    Used to create the name of the chat.
    :return: The name of the chat.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        verbose=True,
        temperature=0.3
    )
    response = llm.invoke(
        f"This is the prompt given by the user : {user_prompt} \n, provide a suitable title for the chat related to this prompt. Provide only the title, \
                          No unnecessary greetings or verbose messages."
    )
    return response.content


async def authenticate_user(user_name: str, password: str):
    """
    Used for checking if the user is a registered user or not.
    :param user_name: Name of the user.
    :param password: Password of the user.
    :return: Boolean verification,
    """
    try:
        stored = await collection.find_one({"user_name": user_name}, {"_id": 0, "password": 1})
        if not stored:
            return False
        return bcrypt.verify(password, stored["password"])  # Verifying the password.
    except Exception:
        return False


def validate_password(password: str):
    """
    Validate password strength.
    :param password: Password to validate
    :return: Tuple of (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one digit"
    if not any(c.isalpha() for c in password):
        return False, "Password must contain at least one letter"
    return True, ""


def get_model_runner(user_name: str) -> RunModel:
    """
    Get or create a cached RunModel instance for a user.
    :param user_name: Name of the user
    :return: RunModel instance
    """
    if user_name not in model_runner_cache:
        model_runner_cache[user_name] = RunModel(db_name=f"VectorDB_{user_name}")
    return model_runner_cache[user_name]


@app.post("/sign_in")
async def sign_in(user_name: str, password: str):
    """
    API endpoint for user registration.
    :param user_name: Name of the user.
    :param password: Password of the user.
    :return: Confirmation message.
    """
    # Checking if the user already exists
    data = await collection.find_one({"user_name": user_name})
    if data:
        raise HTTPException(status_code = 409, detail="User with this user_name already exists.")

    # Validating password strength
    is_valid, error_msg = validate_password(password)
    if not is_valid:
        raise HTTPException(status_code = 400, detail = error_msg)

    # Generating an encryption key and hash password
    key = Fernet.generate_key()
    hashed_password = bcrypt.hash(password)  # Using hashed password for extra safety.

    # Creating user document
    user_doc = {
        "user_name": user_name,
        "password": hashed_password,
        "encryption_key": key,
        "session_ids": [],
        "chat_names": [],
        "chats": [],
        "created_at": datetime.utcnow()
    }

    try:
        await collection.insert_one(user_doc)
        return {"confirmation": "Signed In Successfully. Proceed to Login"}
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Failed to create user: {str(e)}")


@app.post("/login")
async def login(user_name: str, password: str):
    """
    API endpoint for authorising the login.
    :param user_name: The name of the user.
    :param password: The password of the user.
    :return: The confirmation message.
    """
    # Check if user exists
    user = await collection.find_one({"user_name": user_name})
    if not user:
        raise HTTPException(status_code = 401,
                            detail = "Invalid credentials !!! \n Plz Sign in First OR Check the Credentials.")

    # Authenticate user
    is_authenticated = await authenticate_user(user_name, password)
    if not is_authenticated:
        raise HTTPException(status_code = 401,
                            detail = "Invalid credentials !!! \n Plz Sign in First OR Check the Credentials.")

    return {"login_status": "Successful"}


@app.get("/{user_name}/all_sessions")
async def list_sessions(user_name: str):
    """
    Give a list of all the chat sessions created by the user.
    :param user_name: The name of the user.
    :return: The list of session_ids and their respective chat names.
    """
    try:
        # First check if user exists
        user = await collection.find_one({"user_name": user_name})
        if not user:
            raise HTTPException(status_code = 404, detail = "User not found.")

        # Get sessions data
        sessions = await collection.find_one(
            {"user_name": user_name},
            {"_id": 0, "session_ids": 1, "chat_names": 1, "encryption_key": 1}
        )

        if not sessions:
            return {"session_ids": [], "chat_names": []}

        # Decrypting chat names
        key = sessions.get("encryption_key")
        if not key:
            return {"session_ids": sessions.get("session_ids", []), "chat_names": []}

        cipher = Fernet(key)
        chat_names = []

        for encrypted_name in sessions.get("chat_names", []):
            try:
                if encrypted_name:
                    decrypted = cipher.decrypt(encrypted_name)
                    chat_names.append(decrypted.decode('utf-8'))
                else:
                    chat_names.append("Untitled Chat")
            except Exception:
                chat_names.append("Untitled Chat")

        return {
            "session_ids": sessions.get("session_ids", []),
            "chat_names": chat_names
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))


@app.post("/{user_name}/new_session")
async def new_session(user_name: str):
    """
    Creates a new session id for the user to start a new conversation.
    :param user_name: The unique user_name of the user.
    :return: The new session id.
    """
    result = await collection.find_one({"user_name": user_name})
    if not result:
        raise HTTPException(status_code=404, detail="User not found.")

    # Generate unique session ID
    max_attempts = 10
    for _ in range(max_attempts):
        session_id = str(uuid.uuid4())
        # Checking if this session_id already exists for this user
        existing = await collection.find_one(
            {"user_name": user_name, "session_ids": session_id}
        )
        if not existing:
            return {"session_id": session_id}

    raise HTTPException(status_code = 500, detail = "Failed to generate unique session ID")


@app.post("/{user_name}/{session_id}/chat")
async def chat(user_prompt: str, session_id: str, user_name: str, chat_type = "text"):
    """
    API endpoint which enables user to interact with the llm.
    :param user_prompt: The query of the user.
    :param session_id: The chat the user want to ask the query in.
    :param user_name: The name of the user.
    :param chat_type: The type of chat the user want to have. It can be either "text" (by default) or "voice".
    :return: The response given by the llm.
    """
    try:
        # Validating user exists and getting encryption key
        doc = await collection.find_one({"user_name": user_name})
        if not doc:
            raise HTTPException(status_code=404, detail="User not found.")

        key = doc.get("encryption_key")
        if not key:
            raise HTTPException(status_code=500, detail="Encryption key not found.")

        cipher = Fernet(key)

        # Check if this is a new session
        session_ids = doc.get("session_ids", [])
        is_new_session = session_id not in session_ids

        if is_new_session:
            # Create chat name for new session
            chat_name = create_chat_name(user_prompt=user_prompt)
            encrypted_chat_name = cipher.encrypt(chat_name.encode('utf-8'))

            # Initialize new session with empty chat array
            await collection.update_one(
                {"user_name": user_name},
                {
                    "$push": {
                        "session_ids": session_id,
                        "chat_names": encrypted_chat_name,
                        "chats": []  # Initialize empty array for this session's messages
                    }
                }
            )
        else:
            # Ensure chats array has correct length
            session_index = session_ids.index(session_id)
            chats = doc.get("chats", [])

            # If chats array is shorter than session_ids, pad it with empty arrays
            while len(chats) <= session_index:
                await collection.update_one(
                    {"user_name": user_name},
                    {"$push": {"chats": []}}
                )

        # Get model runner and generate response
        model_runner = get_model_runner(user_name)
        response = model_runner.run(
            user_prompt = user_prompt,
            session_id = session_id,
            user_name = user_name
        )

        # Encrypt messages
        encrypted_user_message = cipher.encrypt(user_prompt.encode('utf-8'))
        encrypted_ai_message = cipher.encrypt(response.encode('utf-8'))

        # Create message object
        message_obj = {
            "user_message": encrypted_user_message,
            "ai_message": encrypted_ai_message,
            "timestamp": datetime.utcnow()
        }

        # Find session index and update the specific chat array
        updated_doc = await collection.find_one({"user_name": user_name})
        current_session_ids = updated_doc.get("session_ids", [])

        if session_id in current_session_ids:
            session_index = current_session_ids.index(session_id)

            # Update the specific chat array for this session
            await collection.update_one(
                {"user_name": user_name},
                {"$push": {f"chats.{session_index}": message_obj}}
            )

        return {"response": response}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete_chat/{user_name}/{session_id}")
async def delete_chat(user_name: str, session_id: str):
    """
    Delete a specific chat session for a user.
    :param user_name: Name of the user
    :param session_id: Session ID to delete
    :return: Success message
    """
    # Finding the index of the session to delete
    user_doc = await collection.find_one({"user_name": user_name})
    if not user_doc:
        raise HTTPException(status_code=404, detail="User not found")

    session_ids = user_doc.get("session_ids", [])
    if session_id not in session_ids:
        raise HTTPException(status_code=404, detail="Session ID not found")

    try:
        removed_idx = session_ids.index(session_id)

        # Get current arrays
        chat_names = user_doc.get("chat_names", [])
        chats = user_doc.get("chats", [])

        # Remove elements at the specific index
        new_session_ids = session_ids[:removed_idx] + session_ids[removed_idx + 1:]
        new_chat_names = chat_names[:removed_idx] + chat_names[removed_idx + 1:] if removed_idx < len(
            chat_names) else chat_names
        new_chats = chats[:removed_idx] + chats[removed_idx + 1:] if removed_idx < len(chats) else chats

        # Update with new arrays (atomic operation)
        await collection.update_one(
            {"user_name": user_name},
            {
                "$set": {
                    "session_ids": new_session_ids,
                    "chat_names": new_chat_names,
                    "chats": new_chats
                }
            }
        )

        return {"status": "success", "message": "Chat deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")


@app.get("/{user_name}/{session_id}/messages")
async def get_messages(user_name: str, session_id: str, limit: Optional[int] = 50):
    """
    Get messages for a specific session with pagination.
    :param user_name: Name of the user
    :param session_id: Session ID
    :param limit: Maximum number of messages to return
    :return: List of decrypted messages
    """
    try:
        user_doc = await collection.find_one({"user_name": user_name})
        if not user_doc:
            raise HTTPException(status_code=404, detail="User not found")

        session_ids = user_doc.get("session_ids", [])
        if session_id not in session_ids:
            raise HTTPException(status_code=404, detail="Session not found")

        session_index = session_ids.index(session_id)
        chats = user_doc.get("chats", [])

        if session_index >= len(chats):
            return {"messages": []}

        # Get messages for this session
        session_messages = chats[session_index] if chats[session_index] else []

        # Decrypt messages
        key = user_doc.get("encryption_key")
        if not key:
            return {"messages": []}

        cipher = Fernet(key)
        decrypted_messages = []

        # Get last 'limit' messages
        for msg in session_messages[-limit:]:
            try:
                decrypted_msg = {
                    "user_message": cipher.decrypt(msg["user_message"]).decode('utf-8'),
                    "ai_message": cipher.decrypt(msg["ai_message"]).decode('utf-8'),
                    "timestamp": msg.get("timestamp", "")
                }
                decrypted_messages.append(decrypted_msg)
            except Exception:
                continue

        return {"messages": decrypted_messages}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app = app, port = 8001)