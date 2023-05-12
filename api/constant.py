import json
WETH_ADDRESS_GOERLI = '0xb4fbf271143f4fbf7b91a5ded31805e42b2208d6'
WETH_ADDRESS_BSC_TEST = '0xae13d989daC2f0dEbFf460aC112a837C89BAa7cd'
INFURA_GOERLI_ENDPOINT = 'https://goerli.infura.io/v3/'
with open('./contract-metadatas/cake.factory.json') as cakeFactory:
    data = json.load(cakeFactory)
    CAKE_FACTORY_ADDRESS_GOERLI = data['CAKE_FACTORY_ADDRESS_GOERLI']
    CAKE_FACTORY_ABI_GOERLI = data['CAKE_FACTORY_ABI_GOERLI']

with open('./contract-metadatas/cake.router.json') as cakeRouter:
    data = json.load(cakeRouter)
    CAKE_ROUTER_ADDRESS_GOERLI = data['CAKE_ROUTER_ADDRESS_GOERLI']
    CAKE_ROUTER_ABI_GOERLI = data['CAKE_ROUTER_ABI_GOERLI']

print('haha')
