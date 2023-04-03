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


@router.get('/latest/{address}')
def latestTransactions(address : str):

    addresses = investorDocs.distinct('_id')

    if address not in addresses:
        return {
            'message' : f'{address} not found'
        }
    
    query = {'_id' : { '$eq' : address}}
    projection = {'_id': 1, 'TXs': { '$slice': -3 }}
    # 0x3BC643A841915A267eE067b580BD802a66001C1d
    response = investorDocs.find_one(filter=query,projection=projection)
    response['message'] = 'Successfully'
    return response


