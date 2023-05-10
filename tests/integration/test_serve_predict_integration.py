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


@pytest.fixture()
def client():
    webserver_process = subprocess.Popen(SERVE_COMMAND.split(), cwd=ROOT_DIR)
    sleep(5)
    yield webserver_process
    webserver_process.terminate()


def test_serving_predictions(client):
    body = {'data': get_test_datapoints(1)}
    resp = requests.post(PREDICT_URL, json=body)
    assert resp.status_code == 200
    assert len(resp.json()) == 1
