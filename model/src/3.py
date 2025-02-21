from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np

# Connect to MongoDB
client = MongoClient('mongodb://172.31.30.235:27017')
db = client['chesto']  # replace with your database name
collection = db['samples']

# # Project only the rating field and convert cursor to list
# high_rating_count = collection.count_documents({'rating': {'$gt': 2100}})

# # Get total document count for comparison
# total_count = collection.count_documents({})

# # Calculate percentage
# percentage = (high_rating_count / total_count * 100) if total_count > 0 else 0

# print(f"Documents with rating > 2100: {high_rating_count}")
# print(f"Total documents: {total_count}")
# print(f"Percentage of high-rated documents: {percentage:.2f}%")