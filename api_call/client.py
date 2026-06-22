import os
import requests
from dotenv import load_dotenv
from registry import ACTIONS

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
BASE_URL = "http://localhost:3000/api/v3/apiBuilderService"


def execute(action: str, payload=None, entity_id=None):
    config = ACTIONS[action]

    url = f"{BASE_URL}/{action}"

    if entity_id:
        url += f"/{entity_id}"

    headers = {
        "Cookie": f"AUTH_TOKEN={API_TOKEN}",
        "Content-Type": "application/json",
    }

    method = config["method"]

    if method == "POST":
        r = requests.post(url, json=payload, headers=headers)

    elif method == "PUT":
        r = requests.put(url, json=payload, headers=headers)

    elif method == "DELETE":
        r = requests.delete(url, headers=headers)

    elif method == "GET":
        r = requests.get(url, params=payload, headers=headers)

    else:
        raise ValueError(method)

    return r.status_code, r.text