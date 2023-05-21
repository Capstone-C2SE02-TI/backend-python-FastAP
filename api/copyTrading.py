from pydantic import BaseModel
from api.mongoDB_init import crawlClient
from fastapi import APIRouter, Form
from fastapi import FastAPI
from hexbytes import HexBytes
from api.constant import WETH_ADDRESS_GOERLI, WETH_ADDRESS_BSC_TEST
from api.util import numToBytes, hexToBytes, getSecondUnix, getRandomInfuraKey
from api.web3Utils import *
from api.mongoDB_init import main2Client
from web3.middleware import geth_poa_middleware
import os
from dotenv import load_dotenv
from web3.middleware import construct_sign_and_send_raw_middleware
from eth_account import Account
from eth_account.signers.local import LocalAccount


load_dotenv()

app = FastAPI()

investorDocs = crawlClient['investors']
router = APIRouter(
    prefix='/copyTrading',
)


def getTradingInput(buy_token_address, receiver):
    # return dir(pancakeFactory.functions)

    LPollAddress = pancakeFactory.functions.getPair(
        web3.to_checksum_address(WETH_ADDRESS_BSC_TEST),
        web3.to_checksum_address(buy_token_address)
    ).call()

    if LPollAddress == '0x0000000000000000000000000000000000000000':

        return False, 'This pair didnt exist'

    print("Pair Address", [hexToBytes(WETH_ADDRESS_BSC_TEST[2:]),
                           hexToBytes(buy_token_address[2:])])
    print("Pool Address", LPollAddress)
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
    path = [hexToBytes(WETH_ADDRESS_BSC_TEST[2:]),
            hexToBytes(buy_token_address[2:])]
    print([functionSelector, amountOutMin, mystery,
          to, deadline, pathLen, path[0], path[1]])

    return True, ''.join(
        [functionSelector, amountOutMin, mystery, to, deadline, pathLen, path[0], path[1]])


@router.post("/hash/")
async def getTransactionInput(buy_token_address: str = Form("0xeD24FC36d5Ee211Ea25A80239Fb8C4Cfd80f12Ee"),
                              receiver: str = Form(
                                  "0x72598E10eF4c7C0E651f1eA3CEEe74FCf0A76CF1"),
                              chain_id: int = Form(5)):

    response = {
        'message': 'Success',
        'input': None
    }
    print('-'*70)
    print("Get input for:", buy_token_address, receiver)
    result, input = getTradingInput(buy_token_address, receiver)

    if result:
        response['input'] = input
    else:
        response['message'] = input

    return response
    return '0xced03d77000000000000000000000000eff92a263d31888d860bd50809a8d171709b7b1c000000000000000000000000000000000000000000000000000000000000004000000000000000000000000000000000000000000000000000000000000000e47ff36ab5000000000000000000000000000000000000000000000002b5e3af16b1880000000000000000000000000000000000000000000000000000000000000000008000000000000000000000000072598e10ef4c7c0e651f1ea3ceee74fcf0a76cf2000000000000000000000000000000000000000000000000000000006441492e0000000000000000000000000000000000000000000000000000000000000002000000000000000000000000b4fbf271143f4fbf7b91a5ded31805e42b2208d60000000000000000000000002c974b2d0ba1716e644c1fc59982a89ddd2ff72400000000000000000000000000000000000000000000000000000000'
    # Transfer to array and return it.

# web3 = Web3(Web3.HTTPProvider(
#     "https://data-seed-prebsc-1-s2.binance.org:8545"))
# web3.middleware_onion.inject(geth_poa_middleware, layer=0)
# middle = web3.eth.contract(
#         address=MIDDLE_ADDRESS_BSC_TEST, abi=MIDDLE_ABI_BSC_TEST)


@router.post("/auto")
async def autoTrading(receiver: str = Form("0xeD24FC36d5Ee211Ea25A80239Fb8C4Cfd80f12Ee"),
                      dex_address: str = Form(
                          "0x72598E10eF4c7C0E651f1eA3CEEe74FCf0A76CF2"),
                      input_data: str = Form("0x00"),
                      eth_amount: float = Form(0.001)):

    dex_address = "0x9Ac64Cc6e4415144C455BD8E4837Fea55603e5c3"
    receiver = web3.to_checksum_address(receiver),

    print("dex_address", dex_address)
    print("input_data", input_data)
    print("eth_amount", eth_amount)
    print("receiver", receiver)

    private_key = os.environ.get("private_key")
    account: LocalAccount = Account.from_key(private_key)
    print(account.address, "Signer")
    web3.middleware_onion.add(construct_sign_and_send_raw_middleware(account))
    web3.eth.default_account = account.address
    nonce = web3.eth.get_transaction_count(
        account.address)

    eth_amount = int(eth_amount * 1000000000000000000)
    tx = middle.functions.copyTrading(receiver, dex_address, hexToBytes(input_data), eth_amount).build_transaction({
        'nonce': nonce,
        'gas': 200000,
        'from': account.address,
        'gasPrice': 12345600000,
    })

    print(tx)
    sended_tx = web3.eth.send_transaction(tx)

    print(HexBytes(sended_tx))

    return 0
    # print(sended_tx)
