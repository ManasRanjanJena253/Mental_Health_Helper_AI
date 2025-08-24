import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import asyncio
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGO_URI")
db_name = "Mental_Health_AI_User_Database"

validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_name", "password", "encryption_key"],
        "properties": {
            "user_name": {
                "bsonType": "string",
                "description": "Unique username for the user"
            },
            "password": {
                "bsonType": "string",
                "description": "Hashed password using bcrypt"
            },
            "encryption_key": {
                "bsonType": "binData",
                "description": "Fernet encryption key for encrypting user data"
            },
            "session_ids": {
                "bsonType": "array",
                "description": "Array of session IDs",
                "items": {
                    "bsonType": "string"
                }
            },
            "chat_names": {
                "bsonType": "array",
                "description": "Array of encrypted chat names",
                "items": {
                    "bsonType": "binData"
                }
            },
            "chats": {
                "bsonType": "array",
                "description": "Array of chat sessions",
                "items": {
                    "bsonType": "array",
                    "description": "Messages in a session",
                    "items": {
                        "bsonType": "object",
                        "required": ["user_message", "ai_message"],
                        "properties": {
                            "user_message": {
                                "bsonType": "binData",
                                "description": "Encrypted user message"
                            },
                            "ai_message": {
                                "bsonType": "binData",
                                "description": "Encrypted AI response"
                            },
                            "timestamp": {
                                "bsonType": "date",
                                "description": "Message timestamp"
                            }
                        }
                    }
                }
            },
            "created_at": {
                "bsonType": "date",
                "description": "Account creation timestamp"
            }
        }
    }
}

async def create_async_collection():
    client = AsyncIOMotorClient(uri, server_api = ServerApi("1"))
    try:
        # Use await for async admin command
        await client.admin.command("ping")
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print("Ping failed:", e)
        client.close()
        return

    db = client[db_name]

    collections = await db.list_collection_names()
    if "user_data" not in collections:
        await db.create_collection(
            name="user_data",
            validator=validator,
            validationAction="error"
        )
        print("user_data Collection Created")
    else:
        print("user_data Collection Already Exists")


if __name__ == "__main__":
    asyncio.run(create_async_collection())
