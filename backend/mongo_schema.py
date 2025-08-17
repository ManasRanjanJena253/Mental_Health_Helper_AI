from pymongo import MongoClient

client = MongoClient(port = 27017, host = "localhost")

db_name = "Mental_Health_AI_User_Database"
db = client[db_name]

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
            "password": {"bsonType": "string"},
            "encryption_key": {"bsonType": "binData"}   # Binary data type for storing encryption keys.
        }
    }
}

# Creating the user_data collection
#db.drop_collection("user_data")
collections = db.list_collection_names()
if 'user_data' not in collections:
    db.create_collection(name = 'user_data',
                        validator = validator,
                        validationAction = 'error')
    print('user_data Collection Created')
else :
    print('user_data Collection Already Exists')