from pymongo import MongoClient

client = MongoClient("mongodb+srv://Pavan_Rajkumar:back-bench-kool@pdp.jrsmd.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.get_database('PDP')
records = db.Predictions

def writeToDB(data):
    records.insert_one(data)
    return

def readFromDB():
    data=list(records.find())
    return data

def readSingleFromDB(id):
    datar=records.find_one({'first': id})
    print(id)
    print(datar)
    return datar

def countInDB():
    num= records.count_documents({})
    return num

def countGender():
    gender =[]
    gender.append(['Male',len(list(records.find({ 'gender': 'Male' })))])
    gender.append(['Female',len(list(records.find({ 'gender': 'Female' })))])
    print(records.find())
    # print(records.find({ gender: 'Male' }).count())
    # print(records.find({ gender: 'Female' }).count())
    return gender