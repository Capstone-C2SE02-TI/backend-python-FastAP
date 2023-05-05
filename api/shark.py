from api.mongoDB_init import crawlClient
from fastapi import APIRouter
from fastapi import FastAPI, Form


app = FastAPI()

investorDocs = crawlClient['investors']
router = APIRouter(
    prefix='/shark',
)


# @router.get('/latest/{address}')

@router.get('/token_trading')
def tokenTrading(address: str = "0x72598E10eF4c7C0E651f1eA3CEEe74FCf0A76CF2"):
    addresses = investorDocs.distinct('_id')
    if address not in addresses:
        return {
            'message': f'{address} not found'
        }
    
    query = {'_id': {'$eq': address}}
    projection = {'_id': 1, 'pair_tradings': 1}
    response = investorDocs.find_one(filter=query, projection=projection)
    pair_tradings = response['pair_tradings']

    return pair_tradings



@router.post('/latest_tx/')
def latestTransactions(address: str = Form("0x72598E10eF4c7C0E651f1eA3CEEe74FCf0A76CF2"), 
                       contract_address: str = Form(""), 
                       pages: int = Form(1)):

    # address = "0x5a52E96BAcdaBb82fd05763E25335261B270Efcb"
    # pages = 1
    txPerPages = 10

    addresses = investorDocs.distinct('_id')
    if address not in addresses:
        return {
            'message': f'{address} not found'
        }
    
    query = {'_id': {'$eq': address}}
    projection = {'_id': 1, 'TXs': {
        '$slice': [-txPerPages*(pages), txPerPages*pages]}}
    response = investorDocs.find_one(filter=query, projection=projection)
    TXs = response['TXs']

    print(contract_address, contract_address != '')
    if contract_address != '':
        filteredTXs = []
        for tx in TXs:
            if 'contractAddress' not in tx or tx['contractAddress'].lower() != contract_address.lower():    
                continue
            filteredTXs.append(tx)

        response['TXs'] = filteredTXs

    response['message'] = 'Successfully'
    return response
