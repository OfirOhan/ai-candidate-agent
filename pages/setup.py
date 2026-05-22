import streamlit as st
import os
from rag.ingest import ingest_document
from store.structured import save, load

st.title("Candidate Setup")

st.header("Step 1 — Upload your documents")
uploaded_files = st.file_uploader(
    "Upload CV, grades, certificates (PDF only)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        ingest_document(file_path, "candidate_001")
        st.success(f"Ingested: {uploaded_file.name}")

st.header("Step 2 — Answer key questions")
data = load()

data["full_name"] = st.text_input("Full name", value=data.get("full_name", ""))
data["salary_expectation"] = st.text_input("Salary expectation", value=data.get("salary_expectation", ""))
data["preferred_location"] = st.text_input("Preferred location", value=data.get("preferred_location", ""))
data["availability"] = st.text_input("Available from", value=data.get("availability", ""))
data["work_type"] = st.selectbox("Work type", ["Remote", "Hybrid", "Onsite"], index=0)
data["years_of_experience"] = st.text_input("Years of experience", value=data.get("years_of_experience", ""))
data["open_to_relocation"] = st.selectbox("Open to relocation?", ["Yes", "No"])
data["linkedin"] = st.text_input("LinkedIn URL", value=data.get("linkedin", ""))

if st.button("Save profile"):
    save(data)
    st.success("Profile saved. Share this link with recruiters: http://localhost:8501/recruiter")
