from typing import Union

import pandas as pd
import pytest
import subprocess
from time import sleep
from src.config import ROOT_DIR
from src.serve.app import PREDICT_ROUTE
from src.serve.app import app as flask_app
import requests
from tests.utils import get_test_datapoints

flask_app.config['TESTING'] = True  # not stricly necessary here
SERVE_COMMAND = 'python -m src.main --serve'
SERVER_PORT = 8888
BASE_URL = f'http://localhost:{SERVER_PORT}'
PREDICT_URL = BASE_URL + PREDICT_ROUTE


@pytest.fixture(scope="session")
def client():
    webserver_process = subprocess.Popen(SERVE_COMMAND.split(), cwd=ROOT_DIR)
    sleep(5)
    yield webserver_process
    webserver_process.terminate()


def _send_post(n_size: Union[int, str]):
    if not isinstance(n_size, int):
        body = {'data': "some random string"}
    else:
        body = {'data': get_test_datapoints(n_size)}
    return requests.post(PREDICT_URL, json=body)


@pytest.mark.parametrize("n_size", [1, 10, 100])
def test_serving_predictions(client, n_size):
    resp = _send_post(n_size=n_size)
    assert resp.status_code == 200
    assert len(resp.json()) == n_size


def test_incorrect_serving_prediction(client):
    resp = _send_post("")
    assert resp.status_code == 400

