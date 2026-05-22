import json
import os

DATA_PATH = "./store/data/candidate.json"

DEFAULT_FIELDS = {
    "full_name": "",
    "salary_expectation": "",
    "preferred_location": "",
    "availability": "",
    "work_type": "",       # remote / hybrid / onsite
    "years_of_experience": "",
    "open_to_relocation": "",
    "linkedin": ""
}


def load() -> dict:
    if not os.path.exists(DATA_PATH):
        return DEFAULT_FIELDS.copy()
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def save(data: dict):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_field(field: str) -> str:
    data = load()
    return data.get(field, "Not provided")
