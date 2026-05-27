from agent.llm import LLMClient
from agent.tools import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """
You are an AI representative for a job candidate. 
Your job is to answer recruiter questions accurately and professionally.

Rules:
- For ANY question about the candidate, ALWAYS try get_structured_data first.
  If the answer is not found there, use search_documents as a fallback.
- Never guess or make up information. If you don't find it, say so.
- Keep answers concise and professional.
- Always answer in the same language the recruiter used.

IMPORTANT: When calling tools, pass arguments as plain strings.
Do NOT pass JSON schemas or objects as arguments.
"""

MAX_TOOL_ROUNDS = 3  # safety cap to prevent infinite loops

llm = LLMClient()


def run(conversation_history: list, user_message: str) -> tuple[str, list]:
    """
    Main agent turn.
    Supports multiple sequential tool calls so the LLM can fall back
    from structured data to document search when needed.
    Returns (answer_text, updated_conversation_history)
    """

    # Add new user message to history
    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

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

        # Append the tool interaction so the LLM sees the result
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        messages.append({"role": "tool", "content": tool_result})

        # Loop continues — the LLM will now either call another tool
        # (e.g. fallback to search_documents) or produce a final answer.
    else:
        # Safety: if we exhausted all rounds, do one final call without tools
        response = llm.call(messages=messages)
        answer = response["message"]["content"]

    # Update conversation history
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, conversation_history
