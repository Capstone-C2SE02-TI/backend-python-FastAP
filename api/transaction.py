from fastapi import APIRouter
from fastapi import FastAPI


app = FastAPI()
from api.mongoDB_init import crawlClient

investorDocs = crawlClient['investors']
router = APIRouter(
    prefix ='/tx',
)



# @router.get('/latest/{address}')
@router.get('/latest/')
def latestTransactions(address : str = "0x5a52E96BAcdaBb82fd05763E25335261B270Efcb", pages : int = 1):
    txPerPages = 10
    addresses = investorDocs.distinct('_id')

    if address not in addresses:
        return {
            'message' : f'{address} not found'
        }
    print([ txPerPages*(pages-1)-1,txPerPages*pages])
    query = {'_id' : { '$eq' : address}}
    projection = {'_id': 1, 'TXs': { '$slice': [ -txPerPages*(pages),txPerPages*pages] }}
    # 0x3BC643A841915A267eE067b580BD802a66001C1d
    response = investorDocs.find_one(filter=query,projection=projection)
    response['message'] = 'Successfully'
    return response


