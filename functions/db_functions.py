from pymongo import MongoClient

client = MongoClient("mongodb+srv://Pavan_Rajkumar:back-bench-kool@pdp.jrsmd.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.get_database('PDP')
records = db.Predictions

def writeToDB(data):
    records.insert_one(data)
    return