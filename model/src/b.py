from pymongo import MongoClient
import sys

def main():
   client = MongoClient("mongodb://localhost:27017")
   db = client.get_database("chesto")

   i = 0
   for _ in db.replays.find(
       {"steps.0.sample": {"$exists": True}}, 
       {"steps.0.sample": 1}
   ).batch_size(1000):
       i += 1
       if i % 1000 == 0:
           print(i)
           sys.stdout.flush()

main()