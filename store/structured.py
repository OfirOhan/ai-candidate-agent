import json
import os

DATA_PATH = "./store/data/candidate.json"

DEFAULT_EDUCATION = {
    "degree_title": "",
    "field_of_study": "",
    "institution": "",
    "graduation_year": "",
    "gpa": "",
}

DEFAULT_FIELDS = {
    # Personal Details
    "full_name": "",
    "email_address": "",
    "country_code": "",
    "phone_number": "",
    "linkedin": "",
    "github": "",
    # Education (list of degrees)
    "education": [DEFAULT_EDUCATION.copy()],
    # Experience
    "years_of_experience": "",
    "current_role": "",
    "desired_job_title": "",
    "job_description": "",
    # Job Preferences
    "monthly_salary_expectation": "",
    "preferred_location": "",
    "availability": "",
    "work_type": "",  # Remote / Hybrid / Onsite / No Preference
    "open_to_relocation": "",
}


def load() -> dict:
    if not os.path.exists(DATA_PATH):
        return DEFAULT_FIELDS.copy()
    with open(DATA_PATH) as f:
        data = json.load(f)
    # Migration: convert old flat education fields to list format
    if "education" not in data and "degree_title" in data:
        data["education"] = [
            {
                "degree_title": data.pop("degree_title", ""),
                "field_of_study": data.pop("field_of_study", ""),
                "institution": data.pop("institution", ""),
                "graduation_year": data.pop("graduation_year", ""),
                "gpa": data.pop("gpa", ""),
            }
        ]
    return data


def save(data: dict):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_field(field: str) -> str:
    data = load()

    # Handle education field specially — format all degrees into readable text
    if field == "education":
        entries = data.get("education", [])
        if not entries:
            return "Not provided"
        lines = []
        for _i, edu in enumerate(entries, 1):
            title = edu.get("degree_title", "")
            if not title:
                continue
            parts = [title]
            if edu.get("field_of_study"):
                parts.append(f"in {edu['field_of_study']}")
            if edu.get("institution"):
                parts.append(f"from {edu['institution']}")
            if edu.get("graduation_year"):
                parts.append(f"({edu['graduation_year']})")
            if edu.get("gpa"):
                parts.append(f"- GPA: {edu['gpa']}")
            lines.append(" ".join(parts))
        return "\n".join(lines) if lines else "Not provided"

    # Handle phone_number specially — prepend country code
    if field == "phone_number":
        phone = data.get("phone_number", "Not provided")
        country_code = data.get("country_code", "")
        if country_code and phone != "Not provided":
            return f"{country_code} {phone}"
        return phone

    return data.get(field, "Not provided")
