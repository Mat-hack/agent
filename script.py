import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")

BASE_URL = "http://localhost:3000/api/v3/apiBuilderService"

ACTIONS = {
    # Project
    "createProject": {"method": "POST"},
    "readProject": {"method": "GET", "needs_id": True},
    "updateProject": {"method": "PUT"},
    "deleteProject": {"method": "DELETE", "needs_id": True},
    "listProject": {"method": "GET"},

    # Package
    "createPackage": {"method": "POST"},
    "readPackage": {"method": "GET", "needs_id": True},
    "updatePackage": {"method": "PUT"},
    "deletePackage": {"method": "DELETE", "needs_id": True},
    "listPackage": {"method": "GET"},

    # Service
    "createService": {"method": "POST"},
    "readService": {"method": "GET", "needs_id": True},
    "updateService": {"method": "PUT"},
    "deleteService": {"method": "DELETE", "needs_id": True},
    "listService": {"method": "GET"},

    # ValueType
    "createValueType": {"method": "POST"},
    "readValueType": {"method": "GET", "needs_id": True},
    "updateValueType": {"method": "PUT"},
    "deleteValueType": {"method": "DELETE", "needs_id": True},
    "listValueType": {"method": "GET"},

    # Action
    "createAction": {"method": "POST"},
    "readAction": {"method": "GET", "needs_id": True},
    "updateAction": {"method": "PUT"},
    "deleteAction": {"method": "DELETE", "needs_id": True},
    "listAction": {"method": "GET"},

    # Resource
    "createResource": {"method": "POST"},
    "readResource": {"method": "GET", "needs_id": True},
    "updateResource": {"method": "PUT"},
    "deleteResource": {"method": "DELETE", "needs_id": True},
    "listResource": {"method": "GET"},

    # PropType
    "createPropType": {"method": "POST"},
    "readPropType": {"method": "GET", "needs_id": True},
    "updatePropType": {"method": "PUT"},
    "deletePropType": {"method": "DELETE", "needs_id": True},
    "listPropType": {"method": "GET"},

    # Relation
    "createRelation": {"method": "POST"},
    "readRelation": {"method": "GET", "needs_id": True},
    "updateRelation": {"method": "PUT"},
    "deleteRelation": {"method": "DELETE", "needs_id": True},
    "listRelation": {"method": "GET"},

    # Special
    "generateProject": {"method": "PUT", "needs_id": True},
    "initializeProject": {"method": "PUT", "needs_id": True},
    "backupProject": {"method": "GET", "needs_id": True},
    "restoreProject": {"method": "PUT", "needs_id": True},
}


def call_panel(
    action: str,
    payload: dict | None = None,
):
    if action not in ACTIONS:
        raise ValueError(f"Unsupported action: {action}")

    config = ACTIONS[action]

    url = f"{BASE_URL}/{action}"

    response = requests.request(
        method=config["method"],
        url=url,
        json=payload,
        headers={
            "Cookie": f"AUTH_TOKEN={API_TOKEN}",
            "Content-Type": "application/json",
        },
        timeout=60,
    )

    return response.status_code, response.text

