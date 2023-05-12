from api.constant import *
from api.util import getRandomInfuraKey
import json
from web3 import Web3


def getWeb3Provider():

    infuraKey = getRandomInfuraKey()
    web3 = Web3(Web3.HTTPProvider("https://data-seed-prebsc-1-s2.binance.org:8545"))

    return web3
    
def getPancakeRouterInstance() -> object:

    web3 = getWeb3Provider()

    pancakeRouter = web3.eth.contract(address = CAKE_ROUTER_ADDRESS_GOERLI,abi = CAKE_ROUTER_ABI_GOERLI)

    return pancakeRouter

def getPancakeFactoryInstance() -> object:

    web3 = getWeb3Provider()

    pancakeFactory = web3.eth.contract(address = "0x6725F303b657a9451d8BA641348b6761A6CC7a17",abi = CAKE_FACTORY_ABI_GOERLI)

    return pancakeFactory


if __name__ == '__main__':
    getPancakeRouterInstance()
