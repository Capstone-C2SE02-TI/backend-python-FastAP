from api.constant import *
from api.util import getRandomInfuraKey
import json
from web3 import Web3
from web3.middleware import geth_poa_middleware, construct_sign_and_send_raw_middleware


def getWeb3Provider():

    infuraKey = getRandomInfuraKey()
    web3 = Web3(Web3.HTTPProvider(
        "https://data-seed-prebsc-2-s2.binance.org:8545"))
    web3.middleware_onion.inject(geth_poa_middleware, layer=0)

    return web3


def getPancakeRouterInstance(web3) -> object:

    pancakeRouter = web3.eth.contract(
        address=CAKE_ROUTER_ADDRESS_BSC_TEST, abi=CAKE_ROUTER_ABI_BSC_TEST)

    return pancakeRouter


def getPancakeFactoryInstance(web3) -> object:

    pancakeFactory = web3.eth.contract(
        address="0x6725F303b657a9451d8BA641348b6761A6CC7a17", abi=CAKE_FACTORY_ABI_GOERLI)

    return pancakeFactory


def getMiddleInstance(web3) -> object:

    middle = web3.eth.contract(
        address=MIDDLE_ADDRESS_BSC_TEST, abi=MIDDLE_ABI_BSC_TEST)

    return middle


web3 = getWeb3Provider()
pancakeRouter = getPancakeRouterInstance(web3)
middle = getMiddleInstance(web3)
pancakeFactory = getPancakeFactoryInstance(web3)

if __name__ == '__main__':
    getPancakeRouterInstance()
