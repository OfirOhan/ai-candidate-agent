import streamlit as st
from agent.agent import run

st.title("Chat with the Candidate")
st.caption("Ask anything about the candidate's background, skills, or availability.")

if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation
for msg in st.session_state.history:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
user_input = st.chat_input("Ask a question...")
if user_input:
    with st.spinner("Thinking..."):
        answer, updated_history = run(st.session_state.history.copy(), user_input)
    st.session_state.history = updated_history
    st.rerun()
