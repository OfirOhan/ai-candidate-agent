from agent.llm import LLMClient
from agent.tools import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """
You are an AI representative for a job candidate. 
Your job is to answer recruiter questions accurately and professionally.

Rules:
- Use search_documents for anything about skills, experience, education, or projects.
- Use get_structured_data for salary, location, availability, work type, relocation.
- Use book_interview when the recruiter wants to schedule a meeting.
- Never guess or make up information. If you don't find it, say so.
- Keep answers concise and professional.
- Always answer in the same language the recruiter used.
"""

llm = LLMClient()


def run(conversation_history: list, user_message: str) -> tuple[str, list]:
    """
    Main agent turn.
    Returns (answer_text, updated_conversation_history)
    """

    # Add new user message to history
    conversation_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history

    # First LLM call — may return a tool call or a direct answer
    response = llm.call(messages=messages, tools=TOOL_SCHEMAS)
    message = response["message"]

    # Check if Ollama wants to call a tool
    if message.get("tool_calls"):
        tool_call = message["tool_calls"][0]
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]

        # Execute the tool
        tool_result = execute_tool(tool_name, tool_args)

        # Add tool call + result to messages
        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
        messages.append({
            "role": "tool",
            "content": tool_result
        })

        # Second LLM call — generate final answer using tool result
        final_response = llm.call(messages=messages)
        answer = final_response["message"]["content"]
    else:
        # No tool needed, direct answer
        answer = message["content"]

    # Update conversation history
    conversation_history.append({"role": "assistant", "content": answer})

    return answer, conversation_history
