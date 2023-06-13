from keras.engine.sequential import clear_previously_created_nodes
import motor.motor_tornado
from datetime import datetime
from pymongo import MongoClient

# Requires the PyMongo package.
# https://api.mongodb.com/python/current

client = MongoClient('mongodb://localhost:27017/')
filter={}

result = client['FER']['Expressions'].find(
  filter=filter
)
# MONGODB_URL = "mongodb://localhost:27017/fer?retryWrites=true&w=majority"

def connect_mongo():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['FER']
        print("Connected to MongoDB")
        return db
    except Exception as err:
        print(Exception, err)

def add_data_to_mongo(db, expression):
    try:
        print(f"Inside Mongo {expression}")
        json = {"time": datetime.now(), "expression": expression}
        db.Expressions.insert_one(json)
    except Exception as err:
        print(Exception, err)

def main():
    db = connect_mongo()
    table = db.Expressions
    x = table.delete_many({})
    print("All documents deleted.")

main()
    