import streamlit as st
import os
from rag.ingest import ingest_document
from store.structured import save, load

st.title("Candidate Setup")
st.caption("Fill in your profile so the AI agent can represent you to recruiters.")

data = load()

# ── Section 1: Upload Documents ─────────────────────────────────────────────

st.header("Upload Documents")
st.markdown("Upload your CV, transcripts, certificates, or any supporting documents.")

uploaded_files = st.file_uploader(
    "PDF files only",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)
    for uploaded_file in uploaded_files:
        file_path = f"uploads/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        ingest_document(file_path, "candidate_001")
        st.success(f"Ingested: {uploaded_file.name}")

st.divider()

# ── Section 2: Personal Details ─────────────────────────────────────────────

st.header("Personal Details")

data["full_name"] = st.text_input("Full Name", value=data.get("full_name", ""))
data["email_address"] = st.text_input("Email Address", value=data.get("email_address", ""))

col1, col2 = st.columns([1, 3])
with col1:
    data["country_code"] = st.text_input("Country Code", value=data.get("country_code", "+"))
with col2:
    data["phone_number"] = st.text_input("Phone Number", value=data.get("phone_number", ""))

col3, col4 = st.columns(2)
with col3:
    data["linkedin"] = st.text_input("LinkedIn URL", value=data.get("linkedin", ""))
with col4:
    data["github"] = st.text_input("GitHub URL", value=data.get("github", ""))

st.divider()

# ── Section 3: Education ────────────────────────────────────────────────────

st.header("Education")

degree_options = ["", "High School Diploma", "Associate", "Bachelor's", "Master's", "PhD", "Other"]
current_degree = data.get("degree_title", "")
degree_idx = degree_options.index(current_degree) if current_degree in degree_options else 0

col5, col6 = st.columns(2)
with col5:
    data["degree_title"] = st.selectbox("Degree Title", degree_options, index=degree_idx)
with col6:
    data["field_of_study"] = st.text_input(
        "Field of Study",
        value=data.get("field_of_study", ""),
        placeholder="e.g. Computer Science",
    )

col7, col8 = st.columns(2)
with col7:
    data["institution"] = st.text_input(
        "Institution",
        value=data.get("institution", ""),
        placeholder="e.g. Tel Aviv University",
    )
with col8:
    data["graduation_year"] = st.text_input(
        "Graduation Year",
        value=data.get("graduation_year", ""),
        placeholder="e.g. 2023",
    )

data["gpa"] = st.text_input(
    "GPA",
    value=data.get("gpa", ""),
    placeholder="e.g. 85 / 100",
)

st.divider()

# ── Section 4: Experience ───────────────────────────────────────────────────

st.header("Experience")

col9, col10 = st.columns(2)
with col9:
    data["years_of_experience"] = st.text_input(
        "Years of Experience",
        value=data.get("years_of_experience", ""),
        placeholder="e.g. 3",
    )
with col10:
    data["current_role"] = st.text_input(
        "Current / Last Role",
        value=data.get("current_role", ""),
    placeholder="e.g. Backend Developer at Wix",
)

JOB_DESC_LIMIT = 500
data["job_description"] = st.text_area(
    f"Describe what you do / what you're looking for ({JOB_DESC_LIMIT} char limit)",
    value=data.get("job_description", ""),
    max_chars=JOB_DESC_LIMIT,
    height=120,
    placeholder="e.g. I'm a backend developer with 3 years of experience in Python and cloud infrastructure. Looking for a senior role focused on system design and scalability.",
)

st.divider()

# ── Section 5: Job Preferences ──────────────────────────────────────────────

st.header("Job Preferences")

col11, col12 = st.columns(2)
with col11:
    data["desired_job_title"] = st.text_input(
        "Desired Job Title",
        value=data.get("desired_job_title", ""),
        placeholder="e.g. Machine Learning Engineer",
    )
with col12:
    data["monthly_salary_expectation"] = st.text_input(
        "Monthly Salary Expectation",
        value=data.get("monthly_salary_expectation", ""),
        placeholder="e.g. 25,000 ILS",
    )

col15, col16 = st.columns(2)
with col15:
    data["preferred_location"] = st.text_input(
        "Preferred Location",
        value=data.get("preferred_location", ""),
        placeholder="e.g. Tel Aviv, Israel",
    )
with col16:
    data["availability"] = st.text_input(
        "Available From",
        value=data.get("availability", ""),
        placeholder="e.g. Immediately / July 2026",
    )

col13, col14 = st.columns(2)
with col13:
    work_types = ["Remote", "Hybrid", "Onsite", "No Preference"]
    current_work = data.get("work_type", "Remote")
    work_idx = work_types.index(current_work) if current_work in work_types else 0
    data["work_type"] = st.selectbox("Work Type", work_types, index=work_idx)
with col14:
    reloc_options = ["Yes", "No"]
    current_reloc = data.get("open_to_relocation", "Yes")
    reloc_idx = reloc_options.index(current_reloc) if current_reloc in reloc_options else 0
    data["open_to_relocation"] = st.selectbox("Open to Relocation?", reloc_options, index=reloc_idx)

st.divider()

# ── Save ────────────────────────────────────────────────────────────────────

if st.button("Save Profile", type="primary", use_container_width=True):
    save(data)
    st.success("Profile saved! Share this link with recruiters: http://localhost:8501/recruiter")
