from pymongo import MongoClient
from pprint import pprint

# Connect to MongoDB without auth
client = MongoClient('mongodb://localhost:27017/')
db = client.chesto  # assuming database name is 'pokemon'

pprint(list(db.types.find({}))[0]["_id"])