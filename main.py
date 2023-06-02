from typing import Union
from api import shark, copyTrading, prediction
from fastapi import FastAPI
import uvicorn
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


app.include_router(shark.router, tags=["transaction"])
app.include_router(copyTrading.router, tags=["copyTrading"])
app.include_router(prediction.router, tags=["prediction"])


@app.get("/test")
def read_root():
    return [{"constant": True, "inputs": [{"name": "tokenA", "type": "address"}, {"name": "tokenB", "type": "address"}], "name": "getPair", "outputs": [{"name": "pair", "type": "address"}], "payable": False, "type": "function"}]


if __name__ == '__main__':
    uvicorn.run("main:app", port=8080, host='0.0.0.0', reload=True)
