import subprocess
from time import sleep
from typing import Optional

import requests

from src.config import ROOT_DIR
from src.serve.app import PING_ROUTE, PREDICT_ROUTE
from tests.utils import get_test_datapoints


SERVE_COMMAND = 'python -m src.main --serve'
SERVER_PORT = 5000

BASE_URL = f'http://localhost:{SERVER_PORT}'
PING_URL = BASE_URL + PING_ROUTE
PREDICT_URL = BASE_URL + PREDICT_ROUTE


# ----------------------------------------------------------------------------
# Utils functions used to setup/tear down the server for the test

webserver_process: Optional[subprocess.Popen] = None  # keep a reference


def start_webserver():
    global webserver_process
    webserver_process = subprocess.Popen(SERVE_COMMAND.split(), cwd=ROOT_DIR)
    sleep(5)  # wait for server to accept connections


def kill_webserver():
    if webserver_process is None:
        return
    webserver_process.terminate()

# ----------------------------------------------------------------------------


def main():
    # Since we use Python here, it'd be better to leverage the power of pytest
    # to manage errors and fixtures... This is just an example with the
    # knowledge from day 1
    start_webserver()
    try:
        test_prediction(10)
        test_client_error()
    except Exception as e:
        error = str(e)
    else:
        error = None
    finally:
        kill_webserver()

    print('\n\n\n----------')

    if error is not None:
        print('Test failed: ' + error)
        exit(1)

    print('Test succeeded!')


def test_prediction(prediction_count):
    body = {'data': get_test_datapoints(prediction_count)}

    resp = requests.post(PREDICT_URL, json=body)

    assert resp.status_code == 200
    assert len(resp.json()) == prediction_count



def test_client_error():
    body = {'data': 'This is *not* a properly formatted request body!'}

    resp = requests.post(PREDICT_URL, json=body)

    assert resp.status_code == 400


if __name__ == '__main__':
    main()
