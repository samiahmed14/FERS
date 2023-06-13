from keras.engine.sequential import clear_previously_created_nodes
import motor.motor_tornado
from datetime import datetime

MONGODB_URL = "mongodb://localhost:27017/admin?retryWrites=true&w=majority"

def connect_mongo():
    try:
        client = motor.motor_tornado.MotorClient(MONGODB_URL)
        db = client.FER
        print("Connected to MongoDB")
        return db
    except Exception as err:
        print(Exception, err)

def add_data_to_mongo(db, expression):
    try:
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
    