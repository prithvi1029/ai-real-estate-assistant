# mongo_helper.py

from pymongo import MongoClient

def get_collection(collection_name: str):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["real_estate_assistant"]
    collection = db[collection_name]
    collection.create_index("timestamp")
    return collection
