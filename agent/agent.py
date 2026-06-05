import re

from agent.llm import LLMClient
from agent.tools import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """
You are an AI representative for a job candidate.
Your job is to answer recruiter questions accurately and professionally.

You have two tools:

1. get_structured_data(field) — Returns a single verified field.
   ONLY for these exact fields: full_name, email_address, country_code,
   phone_number, linkedin, github, education, years_of_experience,
   current_role, desired_job_title, job_description,
   monthly_salary_expectation, preferred_location, availability,
   work_type, open_to_relocation.

2. search_documents(query) — Searches the candidate's CV, certificates,
   and project docs using semantic retrieval. Use for skills, projects,
   work history details, certifications, achievements, and any question
   needing context beyond a single field.

Tool selection guidelines:
- If the question asks for a single factual value that matches one of
  the structured fields above, use get_structured_data.
- If the question asks to "describe", "tell me about", "compare",
  "summarize", or "explain" — always use search_documents, even if
  the topic sounds like a structured field. These need rich context.
- If a question requires both a structured fact AND document context
  (e.g., "good fit for a remote role?"), call BOTH tools.
- If get_structured_data returns a brief answer that doesn't fully
  address the question, follow up with search_documents.

Refusal policy:
- If the question is about personal beliefs, religion, politics,
  marital status, health, or other private matters unrelated to
  professional qualifications — politely decline WITHOUT calling
  any tool. Say: "I can only provide information related to the
  candidate's professional qualifications and preferences."

Rules:
- Never guess or fabricate information. If you don't find it, say so.
- Keep answers concise and professional.
- Always answer in the same language the recruiter used.
"""

MAX_TOOL_ROUNDS = 5  # safety cap to prevent infinite loops

llm = LLMClient()


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from the response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def run(conversation_history: list, user_message: str) -> tuple[str, list, list]:
    """
    Main agent turn.
    Supports multiple sequential tool calls so the LLM can fall back
    from structured data to document search when needed.
    Returns (answer_text, updated_conversation_history, tool_trajectory)

    tool_trajectory is a list of dicts:
        [{"tool": "get_structured_data", "args": {"field": "full_name"},
          "result_preview": "full_name: Ofir Ohan"}]
    """

    # Add new user message to history
    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    tool_trajectory = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = llm.call(messages=messages, tools=TOOL_SCHEMAS)
        message = response["message"]
        print("OLLAMA RAW MESSAGE:", message)

        # If no tool call, we have our final answer
        if not message.get("tool_calls"):
            answer = message["content"]
            break

        # Execute each tool call the LLM requested
        tool_call = message["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]

        tool_result = execute_tool(tool_name, tool_args)
        print(f"[Agent] Tool '{tool_name}' returned: {tool_result[:200]}")

        # Record the tool call in the trajectory
        tool_trajectory.append({
            "tool": tool_name,
            "args": tool_args,
            "result_preview": tool_result[:300],
        })

        # Append the tool interaction so the LLM sees the result
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": tool_result})

        # Loop continues — the LLM will now either call another tool
        # (e.g. fallback to search_documents) or produce a final answer.
    else:
        # Safety: if we exhausted all rounds, do one final call without tools
        response = llm.call(messages=messages)
        answer = response["message"]["content"]

    # Strip any Qwen3 thinking blocks from the answer
    answer = _strip_thinking(answer)

    # Update conversation history
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, conversation_history, tool_trajectory
