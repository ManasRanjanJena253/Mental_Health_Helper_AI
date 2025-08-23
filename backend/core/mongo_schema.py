import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import asyncio
client = AsyncIOMotorClient(port = 27017, host = "localhost")

db_name = "Mental_Health_AI_User_Database"
db = client[db_name]

#########################################  For deployment purpose    ###################################################
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGO_URI")

# Create a new client and connect to the server
client_atlas = MongoClient(uri, server_api = ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client_atlas.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

########################################################################################################################

validator = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["user_name", "password"],
        "properties": {
            "user_name": {"bsonType": "string"},
            "session_ids": {"bsonType": "array",
                            "items": {"bsonType": "string"}},
            "chat_names": {"bsonType": "array",
                           "items": {"bsonType": "binData"}},
            "chats": {"bsonType": "array",
                      "items": {"bsonType": "object",
                                "properties": {
                                    "ai_message": {"bsonType": "binData"},
                                    "user_message": {"bsonType": "binData"}
                                }}},
            "password": {"bsonType": "string"},
            "encryption_key": {"bsonType": "binData"}   # Binary data type for storing encryption keys and encrypted data.
        }
    }
}

# Creating the user_data memory_collection
async def create_async_collection():

    #await db.drop_collection("user_data")
    collections = await db.list_collection_names()

    if 'user_data' not in collections:
        await db.create_collection(name = 'user_data',
                            validator = validator,
                            validationAction = 'error')
        print('user_data Collection Created')
    else :
        print('user_data Collection Already Exists')

asyncio.run(create_async_collection())    # For running async functions