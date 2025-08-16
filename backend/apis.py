from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import redis
import uuid
from backend.chat_runner import RunModel

app = FastAPI()

model_runner = RunModel()

client_redis = redis.Redis(host = "localhost", db = 0, port = 1000)

mock_user_db = {"manas": "user123", "daksh": "genai", "aman": "frontend"}

def authenticate_user(user: str, password: str):
    return mock_user_db.get(user) == password

@app.post("/login")
def login(username: str, password: str):
    """
    API endpoint for authorising the login.
    :param username: The name of the user.
    :param password: The password of the user.
    :return: The new session token the user is now associated with for this use.
    """
    if not authenticate_user(username, password):
        raise HTTPException(status_code = 401, detail = "Invalid credentials")

    session_token = str(uuid.uuid4())
    client_redis.set(session_token, username)   # permanent session token tied to the user.
    return {"token": session_token}

@app.get("/sessions")
def list_sessions(token: str):
    """
    Give a list of all the chat sessions created by the user.
    :param token: The unique session token tied to a particular user.
    :return:
    """
    username = client_redis.get(token)
    if not username:
        raise HTTPException(status_code = 401, detail = "Invalid token")

    user_id = username.decode()
    results = model_runner.collection.get(where = {"user_id": user_id})
    session_ids = list({meta["session_id"] for meta in results["metadatas"]})
    return {"sessions": session_ids}

@app.post("/new_session")
def new_session(token: str):
    """
    Creates a new session id for the user to start a new conversation.
    :param token: The unique token generated for the user
    :return: The new session id.
    """
    username = client_redis.get(token)

    if not username:
        raise HTTPException(status_code = 401, detail = "Invalid token")

    session_id = str(uuid.uuid4())
    return {"session_id": session_id}

@app.post("/chat/{session_id}")
def chat(user_prompt: str, session_id: str, token: str):
    """
    API endpoint which enables user to interact with the llm.
    :param user_prompt: The query of the user.
    :param session_id: The chat the user want to ask the query in.
    :param token: The unique token generated for the user.
    :return: The response given by the llm.
    """
    user_id = client_redis.get(token).decode()

    try:
        response = model_runner.run(user_prompt = user_prompt,
                         session_id = session_id,
                         user_id = user_id)
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))

    return {"response": response}

@app.delete("delete/{session_id}")
def delete_chat(session_id: str, token: str):
    """
    API endpoint to delete a particular chat history.
    :param session_id: The chat id that the user want to delete.
    :param token: The unique token generated for the user.
    :return: Confirmation message.
    """
    user_id = client_redis.get(token).decode()

    try:
        model_runner.collection.delete(where = {"session_id": {"$eq": session_id}})
        return {"Confirmation": "Successful"}

    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))