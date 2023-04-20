from pydantic import BaseModel
from api.mongoDB_init import crawlClient
from fastapi import APIRouter
from fastapi import FastAPI
from hexbytes import HexBytes
from api.constant import WETH_ADDRESS_GOERLI
from api.util import numToBytes, hexToBytes, getSecondUnix
from api.web3Utils import getPancakeFactoryInstance, getWeb3Provider

web3 = getWeb3Provider()
app = FastAPI()

investorDocs = crawlClient['investors']
router = APIRouter(
    prefix='/copyTrading',
)


class copyTradingRequest(BaseModel):
    txHash: str
    sender: str | None
    ethSendAmount: float
    tax: float | None


@router.post("/hash/")
async def getTransaction(hash='3', chainId=5):

    # w3 = Web3(Web3.HTTPProvider(f'https://goerli.infura.io/v3/{infura_key}'))
    response = {
        'message' : None,
        'response' : None
    }
    txData = {
        "blockNumber": "17001424",
        "timeStamp": "1680929207",
        "hash": "0xb0fea353c68761285d77d26f118d7a45261f18fe1c0ee1ab8364bbed3ba88529",
        "nonce": "861",
        "blockHash": "0x9d679773e8a264be6cf7de02d774a24f3db0e0e2e91849100859503ad296757e",
        "from": "0x5a52e96bacdabb82fd05763e25335261b270efcb",
        "contractAddress": "0x2c974b2d0ba1716e644c1fc59982a89ddd2ff724",
        "to": "0x28c6c06298d514db089934071355e5743bf21d60",
        "value": "260000000000000000000000",
        "tokenName": "VIB",
        "tokenSymbol": "VIB",
        "tokenDecimal": "18",
        "transactionIndex": "3",
        "gas": "36838",
        "gasPrice": "28942154111",
        "gasUsed": "36838",
        "cumulativeGasUsed": "163648",
        "input": "deprecated",
        "confirmations": "23201"
    }

    pancakeFactory = getPancakeFactoryInstance()
    # return dir(pancakeFactory.functions)
    LPollAddress = pancakeFactory.functions.getPair(
        web3.to_checksum_address(WETH_ADDRESS_GOERLI), 
        web3.to_checksum_address('0x07865c6E87B9F70255377e024ace6630C1Eaa37F')
        ).call()
    
    if LPollAddress == '0x0000000000000000000000000000000000000000':
        response['message'] = 'This pair didnt exist'
        return response

    # Function selector : 0x7ff36ab5
    # Amount out min 000000000000000000000000000000000000000000000000000000013433ba5a
    # 0000000000000000000000000000000000000000000000000000000000000080
    # to 00000000000000000000000072598e10ef4c7c0e651f1ea3ceee74fcf0a76cf2
    # Deadline 000000000000000000000000000000000000000000000000000000006422f5f8
    # pathLen : Array of 2? 0000000000000000000000000000000000000000000000000000000000000002
    # Path 1 000000000000000000000000b4fbf271143f4fbf7b91a5ded31805e42b2208d6
    # Path 2 00000000000000000000000007865c6e87b9f70255377e024ace6630c1eaa37f
    functionSelector = '0x7ff36ab5'
    amountOutMin = numToBytes(0)
    mystery = hexToBytes('80')
    to = hexToBytes('0x72598E10eF4c7C0E651f1eA3CEEe74FCf0A76CF2'[2:])
    deadline = numToBytes(getSecondUnix() + 10000)  # Unix in second
    pathLen = numToBytes(2)
    path = [hexToBytes(WETH_ADDRESS_GOERLI[2:]), hexToBytes('0x07865c6E87B9F70255377e024ace6630C1Eaa37F'[2:])]
    print([functionSelector,amountOutMin,mystery,to,deadline,pathLen,path[0],path[1]])

    return ''.join([functionSelector,amountOutMin,mystery,to,deadline,pathLen,path[0],path[1]])

    return '0xced03d77000000000000000000000000eff92a263d31888d860bd50809a8d171709b7b1c000000000000000000000000000000000000000000000000000000000000004000000000000000000000000000000000000000000000000000000000000000e47ff36ab5000000000000000000000000000000000000000000000002b5e3af16b1880000000000000000000000000000000000000000000000000000000000000000008000000000000000000000000072598e10ef4c7c0e651f1ea3ceee74fcf0a76cf2000000000000000000000000000000000000000000000000000000006441492e0000000000000000000000000000000000000000000000000000000000000002000000000000000000000000b4fbf271143f4fbf7b91a5ded31805e42b2208d60000000000000000000000002c974b2d0ba1716e644c1fc59982a89ddd2ff72400000000000000000000000000000000000000000000000000000000'
    # Transfer to array and return it.


@router.post("/hash/")
async def read_item(request: dict):

    print(request)
    response = {

    }

    return 0
