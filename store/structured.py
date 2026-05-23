import json
import os

DATA_PATH = "./store/data/candidate.json"

DEFAULT_FIELDS = {
    # Personal Details
    "full_name": "",
    "email_address": "",
    "country_code": "",
    "phone_number": "",
    "linkedin": "",
    "github": "",

    # Education
    "degree_title": "",
    "field_of_study": "",
    "institution": "",
    "graduation_year": "",
    "gpa": "",

    # Experience
    "years_of_experience": "",
    "current_role": "",
    "desired_job_title": "",
    "job_description": "",

    # Job Preferences
    "monthly_salary_expectation": "",
    "preferred_location": "",
    "availability": "",
    "work_type": "",         # Remote / Hybrid / Onsite / No Preference
    "open_to_relocation": "",
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
