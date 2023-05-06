import requests

body = {
    'address' : 'a',
    'contract_address' : '',
    'pages' : 1
}

import json
body = json.load(body)
# print(requests.post('http://127.0.0.1:8000/shark/latest_tx/', data=body).request.prepare_headers)


def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in 
    this function because it is programmed to be pretty 
    printed and may differ from the actual request.
    """
    print('{}\n{}\r\n{}\r\n\r\n{}'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
        req.body,
    ))

response = requests.post('http://127.0.0.1:8000/shark/latest_tx/', json=body)
print(response.text)
pretty_print_POST(response.request)