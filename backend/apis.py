import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import uuid
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from backend.chat_runner import RunModel
from mongo_schema import db
from passlib.hash import bcrypt
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet

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

app = FastAPI()

collection = db["user_data"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # list of origins; use ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],            # allow all HTTP methods
    allow_headers=["*"],            # allow all headers
)

def create_chat_name(user_prompt: str):
    """
    Used to create the name of the chat.
    :return: The name of the chat.
    """
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash",
                                         google_api_key = os.getenv("GOOGLE_API_KEY"),
                                         verbose = True,
                                         temperature = 0.3)
    response = llm.invoke(f"This is the prompt given by the user {user_prompt}, provide a suitable title for the chat related to this prompt. Provide only the title, \
                          No unnecessary greetings or verbose messages.")

    return response.content

def authenticate_user(user_name: str, password: str):
    """
    Used for checking if the user is a registered user or not.
    :param user_name: Name of the user.
    :param password: Password of the user.
    :return: Boolean verification,
    """
    stored = collection.find_one({"user_name": user_name}, {"_id": 0, "password": 1})
    return bcrypt.verify(password, stored["password"])   # Verifying the password.

@app.post("/sign_in")
def sign_in(user_name: str, password: str):
    data = collection.find_one({"user_name": user_name})
    if data:
        raise HTTPException(status_code = 409, detail = "User with this user_name already exists.")

    key = Fernet.generate_key()
    hashed_password = bcrypt.hash(password)   # Using hashed password for extra safety.
    collection.insert_one({"user_name": user_name, "password": hashed_password, "encryption_key": key})
    return {"Confirmation": "Signed In Successfully. Proceed to Login"}

@app.post("/login")
def login(user_name: str, password: str):
    """
    API endpoint for authorising the login.
    :param user_name: The name of the user.
    :param password: The password of the user.
    :return: The confirmation message.
    """
    user = collection.find_one({"user_name": user_name})
    if not user and not authenticate_user(user_name, password):
        raise HTTPException(status_code = 401, detail = "Invalid credentials !!! \n Plz Sign in First OR Check the Credentials.")

    return {"login_status": "Successful"}

@app.get("/{user_name}/all_sessions")
def list_sessions(user_name : str):
    """
    Give a list of all the chat sessions created by the user.
    :param user_name: The name of the user.
    :return: The list of session_ids and their respective chat names.
    """
    try:
        sessions = collection.find_one({"user_name": user_name}, {"_id": 0, "session_ids": 1, "chat_names": 1})
    except Exception as e:
        raise HTTPException(status_code = 401, detail = str(e))

    key = collection.find_one({"user_name": user_name}, {"_id": 0, "encryption_key": 1})["encryption_key"]
    cypher = Fernet(key = key)
    chat_names = [cypher.decrypt(k) for k in (sessions["chat_names"])]
    return {"session_ids": sessions["session_ids"], "chat_names": chat_names}

@app.post("/{user_name}/new_session")
def new_session(user_name: str):
    """
    Creates a new session id for the user to start a new conversation.
    :param user_name: The unique user_name of the user.
    :return: The new session id.
    """
    result = collection.find_one({"user_name": user_name})
    if result:
        session_id = str(uuid.uuid4())
        return {"session_id": session_id}
    else:
        raise HTTPException(status_code = 404, detail = "User not found.")

@app.post("/{user_name}/{session_id}/chat")
def chat(user_prompt: str, session_id: str, user_name: str):
    """
    API endpoint which enables user to interact with the llm.
    :param user_prompt: The query of the user.
    :param session_id: The chat the user want to ask the query in.
    :param user_name: The name of the user.
    :return: The response given by the llm.
    """
    model_runner = RunModel(db_name=f"VectorDB_{user_name}")

    try:
        session_id_check = collection.find_one({"user_name": user_name, "session_ids": session_id})
        if not session_id_check:  # If the session id is new.
            chat_name = create_chat_name(user_prompt=user_prompt)
            key = collection.find_one({"user_name": user_name})["encryption_key"]
            cypher = Fernet(key)
            encrypted_chat_name = cypher.encrypt(chat_name.encode())
            collection.update_one({"user_name": user_name}, {"$push": {"chat_names": encrypted_chat_name, "session_ids": session_id}})

        response = model_runner.run(user_prompt = user_prompt,
                         session_id = session_id,
                         user_name = user_name)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

    return {"response": response}

@app.delete("/{user_name}/{session_id}/delete")
def delete_chat(user_name: str, session_id: str):
    """
    API endpoint to delete a particular chat history.
    :param session_id: The chat id that the user want to delete.
    :param user_name: The name of the user.
    :return: Confirmation message.
    """
    model_runner = RunModel(db_name=f"VectorDB_{user_name}")
    try:
        model_runner.collection.delete(where = {"session_id": {"$eq": session_id}})
        removed_session_idx = collection.find_one({"user_name": user_name}, {"_id": 0, "session_ids": 1})["session_ids"].index(session_id)

        # Removing the specified session_id from mongodb
        collection.update_one({"user_name": user_name}, {"$pull": {"session_ids.items": session_id}})  # $pull is used to remove an element.

        # Setting the chat_name at the removed session_id index to be null
        collection.update_one({"user_name": user_name}, {"$unset": {f"chat_names.{removed_session_idx}": 1}})

        # Removing the null element from the chat_names
        collection.update_one(({"user_name": user_name}, {"$pull": {"chat_names": None}}))

        return {"Confirmation": "Successful"}

    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

if __name__ == "__main__":
    uvicorn.run(port = 8001, app = app)