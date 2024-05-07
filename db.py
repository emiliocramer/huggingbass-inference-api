import os
from pymongo import MongoClient

# Connect to MongoDB
MONGO_URI = os.environ.get('MONGO_URI')
client = MongoClient(MONGO_URI)

# Get the database and collection
db = client.sourcefuldb
songs_collection = db.songs
models_collection = db.models
reference_artists_collection = db.referenceartists
