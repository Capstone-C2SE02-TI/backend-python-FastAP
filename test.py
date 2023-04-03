import requests

tusdt = "0x07865c6E87B9F70255377e024ace6630C1Eaa37F"
weth = "0xB4FBF271143F4FBf7B91A5ded31805e42b2208d6"
amount = int(10**5)
response = requests.get(f"https://goerli.api.0x.org/swap/v1/quote?buyToken={tusdt}&sellToken={weth}&buyAmount={amount}", timeout=5)

json_data = response.text
print(json_data)
