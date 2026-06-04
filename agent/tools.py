from rag.retriever import retrieve
from store.structured import get_field

CANDIDATE_ID = "candidate_001"  # later: dynamic per candidate

# Store the last retrieval metadata for evaluation access
_last_retrieval_meta = {}

# -- Tool schemas (sent to Ollama) ------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_structured_data",
            "description": (
                "Get fixed, guaranteed-accurate candidate information. "
                "ONLY use this tool for the following fields:\n"
                "  Personal: full_name, email_address, country_code, phone_number, linkedin, github\n"
                "  Education: education (returns all degrees, fields of study, institutions, and GPAs)\n"
                "  Experience: years_of_experience, current_role, desired_job_title, job_description\n"
                "  Preferences: monthly_salary_expectation, preferred_location, availability, work_type, open_to_relocation\n"
                "If the question does not match one of these fields, "
                "use search_documents instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "One of: full_name, email_address, country_code, "
                            "phone_number, linkedin, github, education, "
                            "years_of_experience, current_role, desired_job_title, "
                            "job_description, monthly_salary_expectation, "
                            "preferred_location, availability, work_type, open_to_relocation"
                        )
                    }
                },
                "required": ["field"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the candidate's uploaded documents (CV, grades, certificates). "
                "Use this for ANY question about the candidate that is not covered "
                "by get_structured_data — including skills, experience, education, "
                "projects, contact details, technologies, certifications, and more."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# -- Tool execution functions ------------------------------------------------


def search_documents(**kwargs) -> str:
    global _last_retrieval_meta
    # Accept any argument name the LLM uses (query, field, etc.)
    query = kwargs.get("query", next(iter(kwargs.values()), ""))
    # Auto-recover if the LLM passes a dict instead of a string
    if isinstance(query, dict):
        query = query.get("query", query.get("description", str(query)))
    result = retrieve(str(query), CANDIDATE_ID)
    chunks = result["chunks"]
    # Store retrieval metadata for evaluation
    _last_retrieval_meta = {
        "route": result["route"],
        "expanded_queries": result["expanded_queries"],
        "chunks": chunks,
    }
    if not chunks:
        return "No relevant information found in documents."
    return "\n\n".join(chunks)


def get_structured_data(**kwargs) -> str:
    # Accept any argument name the LLM uses (field, query, etc.)
    field = kwargs.get("field", next(iter(kwargs.values()), ""))
    # Auto-recover if the LLM passes a dict instead of a string
    if isinstance(field, dict):
        raw = field.get("field", field.get("description", str(field)))
        field = raw.split(":")[0].strip()
    value = get_field(str(field))
    return f"{field}: {value}"


def get_last_retrieval_meta() -> dict:
    """Return the metadata from the last search_documents call."""
    return _last_retrieval_meta.copy()


# -- Dispatcher --------------------------------------------------------------

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "get_structured_data": get_structured_data,
}


def execute_tool(name: str, arguments: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    return func(**arguments)
