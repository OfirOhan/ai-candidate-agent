from rag.retriever import retrieve
from store.structured import get_field

CANDIDATE_ID = "candidate_001"  # later: dynamic per candidate

# -- Tool schemas (sent to Ollama) ------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the candidate's uploaded documents (CV, grades, certificates). "
                "Use this when asked about skills, work experience, education, projects, "
                "technologies, or anything that would appear in a resume."
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_structured_data",
            "description": (
                "Get fixed, guaranteed-accurate candidate information. "
                "Use this when asked about salary expectations, location preference, "
                "availability to start, work type (remote/hybrid/onsite), "
                "years of experience, or willingness to relocate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "One of: salary_expectation, preferred_location, "
                            "availability, work_type, years_of_experience, "
                            "open_to_relocation, full_name, linkedin"
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
            "name": "book_interview",
            "description": (
                "Handle interview scheduling. Use this when the recruiter wants to "
                "set up a meeting, check availability for a specific date, or "
                "book an interview slot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "The requested date or time slot"
                    }
                },
                "required": ["date"]
            }
        }
    }
]

# -- Tool execution functions ------------------------------------------------


def search_documents(query: str) -> str:
    chunks = retrieve(query, CANDIDATE_ID)
    if not chunks:
        return "No relevant information found in documents."
    return "\n\n".join(chunks)


def get_structured_data(field: str) -> str:
    value = get_field(field)
    return f"{field}: {value}"


def book_interview(date: str) -> str:
    # MVP: mock response. Later: integrate Calendly API.
    return (
        f"A meeting request has been noted for {date}. "
        "The candidate will confirm via email shortly."
    )


# -- Dispatcher --------------------------------------------------------------

TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "get_structured_data": get_structured_data,
    "book_interview": book_interview
}


def execute_tool(name: str, arguments: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    return func(**arguments)
