from fastapi import APIRouter, Form
from fastapi import FastAPI
from api.mongoDB_init import crawlClient
from api.trading_services.predict import predict_service
import matplotlib.pyplot as plt
app = FastAPI()
router = APIRouter(
    prefix='/prediction',
)

coinDocs = crawlClient['coins']


@router.get("/{symbol}")
def predict_symbol(symbol: str):
    symbol = symbol.lower()

    query = {'symbol': {'$eq': symbol}}
    projection = {'_id': 1, 'prices.daily': 1, 'symbol': 1}
    coin = coinDocs.find_one(filter=query, projection=projection)

    response = {
        "message": None,
        "status": False
    }

    if 'symbol' not in coin:
        response['message'] = 'Symbol not found'
        return response
    if 'prices' not in coin or 'daily' not in coin['prices']:
        response['message'] = 'Symbol not found'
        return response

    response['status'] = True

    preds = predict_service.get_predict(coin['prices']['daily'])
    preds = preds.to_dict('records')

    prediction = [{pred['timestamp']: {
        'price': pred['price'],
        'signal': pred['signal'],
        'history': pred['history'],
        'rawMoney': pred['rawMoney'],
    }} for pred in preds]
    
    response['prediction'] = prediction
    response['symbol'] = coin['symbol']
    return response
