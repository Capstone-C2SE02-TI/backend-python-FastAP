from pydantic import BaseModel
from api.mongoDB_init import crawlClient
from fastapi import APIRouter, Form
from fastapi import FastAPI
from hexbytes import HexBytes
from api.constant import WETH_ADDRESS_GOERLI
from api.util import numToBytes, hexToBytes, getSecondUnix, getRandomInfuraKey
from api.web3Utils import getPancakeFactoryInstance, getWeb3Provider

web3 = getWeb3Provider()
app = FastAPI()

investorDocs = crawlClient['investors']
router = APIRouter(
    prefix='/copyTrading',
)



@router.post("/hash/")
async def getTransactionInput(buy_token_address : str = Form("0x07865c6e87b9f70255377e024ace6630c1eaa37f"),
                         receiver : str = Form("0x72598E10eF4c7C0E651f1eA3CEEe74FCf0A76CF2"),
                         chain_id : int = Form(5)):

    response = {
        'message' : 'Success',
        'input' : None
    }
  
    # pancakeFactory = getPancakeFactoryInstance()
    # # return dir(pancakeFactory.functions)
    # LPollAddress = pancakeFactory.functions.getPair(
    #     web3.to_checksum_address(WETH_ADDRESS_GOERLI), 
    #     web3.to_checksum_address(buy_token_address)
    #     ).call()
    
    # if LPollAddress == '0x0000000000000000000000000000000000000000':
    #     response['message'] = 'This pair didnt exist'
    #     return response

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
    to = hexToBytes(receiver[2:])
    deadline = numToBytes(getSecondUnix() + 10000)  # Unix in second
    pathLen = numToBytes(2)
    # path = [hexToBytes(WETH_ADDRESS_GOERLI[2:]), hexToBytes('0x07865c6E87B9F70255377e024ace6630C1Eaa37F'[2:])]
    path = [hexToBytes("0xae13d989daC2f0dEbFf460aC112a837C89BAa7cd"[2:]), hexToBytes('0xeD24FC36d5Ee211Ea25A80239Fb8C4Cfd80f12Ee'[2:])]
    print([functionSelector,amountOutMin,mystery,to,deadline,pathLen,path[0],path[1]])

    response['input'] = ''.join([functionSelector,amountOutMin,mystery,to,deadline,pathLen,path[0],path[1]])
    return response
    return '0xced03d77000000000000000000000000eff92a263d31888d860bd50809a8d171709b7b1c000000000000000000000000000000000000000000000000000000000000004000000000000000000000000000000000000000000000000000000000000000e47ff36ab5000000000000000000000000000000000000000000000002b5e3af16b1880000000000000000000000000000000000000000000000000000000000000000008000000000000000000000000072598e10ef4c7c0e651f1ea3ceee74fcf0a76cf2000000000000000000000000000000000000000000000000000000006441492e0000000000000000000000000000000000000000000000000000000000000002000000000000000000000000b4fbf271143f4fbf7b91a5ded31805e42b2208d60000000000000000000000002c974b2d0ba1716e644c1fc59982a89ddd2ff72400000000000000000000000000000000000000000000000000000000'
    # Transfer to array and return it.



    

