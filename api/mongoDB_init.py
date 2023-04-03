from fastapi import APIRouter
import pymongo
from dotenv import load_dotenv
import os
from urllib.parse import quote_plus
from fastapi import FastAPI
app = FastAPI()
load_dotenv()

mongodb_password = os.environ['mongodb_password']
mongodb_password = quote_plus(mongodb_password)
connect_string = f'mongodb+srv://hdttuan:{mongodb_password}@cluster0.h09da4d.mongodb.net/?retryWrites=true&w=majority'
crawlClient = pymongo.MongoClient(connect_string)
crawlClient = crawlClient['TRACKINGINVESTMENT_CRAWL']


investorDocs = crawlClient['investors']
router = APIRouter(
    prefix ='/add',
)

address =  "0x3BC643A841915A267eE067b580BD802a66001C1d"

# query = {'_id' : { '$eq' : address}}
# projection = {'_id': 1}
# response = investorDocs.find_one(query=query,projection=projection)
# print('lmao')
# print(response)

print(investorDocs.distinct('_id'))
