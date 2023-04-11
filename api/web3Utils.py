from api.constant import *
from api.util import getRandomInfuraKey
import json
from web3 import Web3


def getWeb3Provider():

    infuraKey = getRandomInfuraKey()
    web3 = Web3(Web3.HTTPProvider(f'{INFURA_GOERLI_ENDPOINT}{infuraKey}'))

    return web3
    
def getPancakeRouterInstance() -> object:

    web3 = getWeb3Provider()

    pancakeRouter = web3.eth.contract(address = CAKE_ROUTER_ADDRESS_GOERLI,abi = CAKE_ROUTER_ABI_GOERLI)

    return pancakeRouter

def getPancakeFactoryInstance() -> object:

    web3 = getWeb3Provider()

    pancakeFactory = web3.eth.contract(address = CAKE_FACTORY_ADDRESS_GOERLI,abi = CAKE_FACTORY_ABI_GOERLI)

    return pancakeFactory


if __name__ == '__main__':
    getPancakeRouterInstance()
