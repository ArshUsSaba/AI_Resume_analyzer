import streamlit as st
import os
import tempfile
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from src.utils import get_gpu_status, log_info, log_error
from src.pdf_reader import extract_text_from_pdf
from src.embeddings import ResumeMatcher
from src.skill_analyzer import analyze_skills
from src.recommender import CareerRecommender

# Set page configuration
st.set_page_config(
    page_title="AI Resume Analyzer and Career Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
custom_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');

/* Main Fonts */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Outfit', sans-serif;
}

/* Gradient Header */
.main-header {
    background: linear-gradient(135deg, #1e0034 0%, #001233 50%, #001e1d 100%);
    padding: 2.5rem;
    border-radius: 16px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.main-header h1 {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 2.6rem !important;
    background: linear-gradient(90deg, #00f2fe 0%, #4facfe 50%, #00f2fe 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.main-header p {
    font-size: 1.1rem;
    opacity: 0.85;
}

/* Sidebar Custom Styling */
[data-testid="stSidebar"] {
    background-color: #0b0f19;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Glassmorphism Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
    margin-bottom: 1.5rem;
}

/* KPI Score Card */
.score-container {
    text-align: center;
    padding: 1rem;
}
.score-number {
    font-size: 3.5rem;
    font-weight: 700;
    font-family: 'Outfit', sans-serif;
    line-height: 1;
    margin-bottom: 0.5rem;
}

/* Badges for Skills */
.skill-badge-match {
    background-color: rgba(0, 242, 254, 0.12);
    color: #00f2fe;
    border: 1px solid rgba(0, 242, 254, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-block;
    margin: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}
.skill-badge-missing {
    background-color: rgba(255, 75, 75, 0.12);
    color: #ff4b4b;
    border: 1px solid rgba(255, 75, 75, 0.3);
    padding: 4px 10px;
    border-radius: 20px;
    display: inline-block;
    margin: 4px;
    font-size: 0.85rem;
    font-weight: 600;
}

/* Modern buttons */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    color: #0c1020;
    font-weight: 600;
    border: none;
    padding: 0.6rem 2rem;
    border-radius: 8px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 242, 254, 0.25);
    width: 100%;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 242, 254, 0.4);
    color: #0c1020;
}

/* Secondary Actions */
.secondary-btn button {
    background-color: transparent !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
.secondary-btn button:hover {
    border-color: #00f2fe !important;
    color: #00f2fe !important;
}

/* Hardware Info styling */
.hardware-panel {
    background-color: #0a0d16;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 1rem;
}
.pulse-badge {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: #00f2fe;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px #00f2fe;
}
.pulse-badge-offline {
    display: inline-block;
    width: 10px;
    height: 10px;
    background-color: #ff9f1c;
    border-radius: 50%;
    margin-right: 8px;
    box-shadow: 0 0 8px #ff9f1c;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ----------------- SESSION STATE & INITIALIZATION -----------------
if "resume_text" not in st.session_state:
    st.session_state.resume_text = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "resume_filename" not in st.session_state:
    st.session_state.resume_filename = ""

# Load models with caching
@st.cache_resource(show_spinner="Loading Sentence Transformer Embeddings model...")
def load_matcher_model():
    return ResumeMatcher()

@st.cache_resource(show_spinner="Loading FLAN-T5 Career Assistant LLM model (can take up to 2 minutes on CPU)...")
def load_recommender_model(use_llm):
    return CareerRecommender(use_llm=use_llm)

# ----------------- SIDEBAR: HARDWARE PANEL -----------------
st.sidebar.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=70)
st.sidebar.title("AI Resume Analyzer Settings")

# LLM Mode Toggle
st.sidebar.subheader("AI Engine settings")
use_transformer_llm = st.sidebar.toggle("Enable FLAN-T5 LLM Generator", value=False, 
                                        help="Toggle ON to load the 1GB FLAN-T5-base model for deep NLP generation. Toggle OFF for instant rule-based recommendations (best for low memory devices).")

# Hardware Status Card
gpu_info = get_gpu_status()
st.sidebar.subheader("Hardware Acceleration")

status_badge = '<span class="pulse-badge"></span>' if gpu_info["available"] else '<span class="pulse-badge-offline"></span>'
status_label = "Active (CUDA)" if gpu_info["available"] else "Offline (CPU)"

st.sidebar.markdown(
    f"""
    <div class="hardware-panel">
        <p style="margin:0 0 8px 0; font-size:0.9rem;"><strong>GPU Acceleration:</strong> {status_badge} {status_label}</p>
        <p style="margin:0 0 4px 0; font-size:0.8rem; color:#888;"><strong>Device Type:</strong> {gpu_info["device_name"]}</p>
        <p style="margin:0 0 4px 0; font-size:0.8rem; color:#888;"><strong>VRAM Allocated:</strong> {gpu_info["vram_allocated_gb"]} GB</p>
        <p style="margin:0; font-size:0.8rem; color:#888;"><strong>VRAM Reserved:</strong> {gpu_info["vram_reserved_gb"]} GB</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Colab Note:** When running on Google Colab with T4 GPU, enabling the FLAN-T5 model provides high-quality custom recommendations with ~1-3s latency. Locally on CPU, it might take 10-30s."
)

# ----------------- MAIN LAYOUT -----------------
st.markdown(
    """
    <div class="main-header">
        <h1>AI Resume Analyzer and Career Assistant</h1>
        <p>NLP-Powered Resume Parsing, Skill Gap Analysis & LLM-Generated Interview Prep (NVIDIA AI Capstone)</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------- SAMPLE DATA HANDLER -----------------
col_setup_1, col_setup_2 = st.columns([1, 4])
with col_setup_1:
    if st.button("📂 Load Sample Data", help="Load built-in resume and job description to test instantly"):
        sample_jd_path = "sample_data/sample_job_description.txt"
        sample_resume_path = "sample_data/sample_resume.pdf"
        
        if os.path.exists(sample_jd_path) and os.path.exists(sample_resume_path):
            try:
                # Load job description
                with open(sample_jd_path, "r", encoding="utf-8") as f:
                    st.session_state.job_desc = f.read()
                
                # Extract text from sample resume PDF
                st.session_state.resume_text = extract_text_from_pdf(sample_resume_path)
                st.session_state.resume_filename = "sample_resume.pdf"
                st.session_state.analysis_results = None # Reset previous results
                st.success("Sample data loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading sample data: {e}")
        else:
            st.warning("Sample data files not found. Run generate_samples.py first.")

# ----------------- STEP 1: UPLOAD & INPUT -----------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 style="margin-top:0;">1. Candidate Resume</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Resume (PDF format)", type=["pdf"], key="uploader")
    
    if uploaded_file is not None:
        if st.session_state.resume_filename != uploaded_file.name:
            # New file uploaded
            with st.spinner("Extracting text from uploaded PDF..."):
                try:
                    # Write to a temporary file because extract_text_from_pdf expects path or file-like object
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    st.session_state.resume_text = extract_text_from_pdf(tmp_path)
                    st.session_state.resume_filename = uploaded_file.name
                    st.session_state.analysis_results = None # Reset previous results
                    os.unlink(tmp_path)
                    st.success("Text extracted successfully!")
                except Exception as e:
                    st.error(f"Failed to read PDF: {e}")
                    
    # Display extracted resume text summary if available
    if st.session_state.resume_text:
        with st.expander(f"View Extracted Text Summary ({st.session_state.resume_filename})"):
            st.text_area("Extracted Plain Text", st.session_state.resume_text, height=200, disabled=True)

with col2:
    st.markdown('<h3 style="margin-top:0;">2. Target Job Description</h3>', unsafe_allow_html=True)
    job_desc_input = st.text_area(
        "Paste the Job Description here...",
        value=st.session_state.job_desc,
        placeholder="Requirements, Responsibilities, and Technical Stack...",
        height=225,
        key="jd_input"
    )
    if job_desc_input != st.session_state.job_desc:
        st.session_state.job_desc = job_desc_input
        st.session_state.analysis_results = None # Reset results if JD changes

# ----------------- ANALYZE TRIGGER -----------------
st.markdown("---")
analyze_button_col = st.columns([1, 2, 1])
with analyze_button_col[1]:
    analyze_clicked = st.button("🚀 Analyze Resume & Match Fit", use_container_width=True)

if analyze_clicked:
    if not st.session_state.resume_text:
        st.error("Please upload a resume PDF or load the sample data first.")
    elif not st.session_state.job_desc:
        st.error("Please enter a target Job Description.")
    else:
        # Run analysis pipeline
        with st.spinner("Processing documents with HuggingFace NLP models..."):
            try:
                # 1. Initialize Matcher & compute embeddings cosine similarity
                matcher = load_matcher_model()
                match_score = matcher.compute_similarity(
                    st.session_state.resume_text, 
                    st.session_state.job_desc
                )
                
                # 2. Skill parsing and taxonomy matching
                skills_report = analyze_skills(
                    st.session_state.resume_text, 
                    st.session_state.job_desc
                )
                
                # 3. Generate recommendations & interview questions
                recommender = load_recommender_model(use_llm=use_transformer_llm)
                
                # Generate tips
                improvement_tips = recommender.generate_improvement_tips(
                    skills_report["missing_skills"],
                    skills_report["matching_skills"]
                )
                
                # Generate wording modifications
                wording_suggestions = recommender.generate_wording_suggestions(
                    skills_report["missing_skills"]
                )
                
                # Generate interview questions
                interview_kit = recommender.generate_interview_questions(
                    skills_report["matching_skills"],
                    skills_report["missing_skills"]
                )
                
                # Store in session state
                st.session_state.analysis_results = {
                    "match_score": match_score,
                    "skills_match_score": skills_report["skills_match_score"],
                    "matching_skills": skills_report["matching_skills"],
                    "missing_skills": skills_report["missing_skills"],
                    "candidate_skills": skills_report["candidate_skills"],
                    "improvement_tips": improvement_tips,
                    "wording_suggestions": wording_suggestions,
                    "interview_kit": interview_kit
                }
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis pipeline crashed: {e}")
                log_error("Analysis crashed", e)

# ----------------- STEP 2: DISPLAY RESULTS DASHBOARD -----------------
if st.session_state.analysis_results is not None:
    res = st.session_state.analysis_results
    
    st.markdown("## Analysis Results & Insights")
    
    # Grid Layout: 3 Columns (Score Card, Skills Match, and Candidate Strengths)
    col_res_1, col_res_2 = st.columns([1, 2])
    
    with col_res_1:
        # Score Circular/Box Indicator
        score = res["match_score"]
        
        # Color coding score
        if score >= 80:
            score_color = "#00f2fe" # Cyan/Blue
            score_badge = "Excellent Fit"
        elif score >= 60:
            score_color = "#ff9f1c" # Amber
            score_badge = "Moderate Fit"
        else:
            score_color = "#ff4b4b" # Red
            score_badge = "Needs Improvement"
            
        st.markdown(
            f"""
            <div class="glass-card score-container">
                <p style="font-size:1.1rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; color:#888;">Semantic Match Score</p>
                <div class="score-number" style="color:{score_color};">{score}%</div>
                <p style="font-size:0.9rem; font-weight:600; color:{score_color}; margin-top:8px;">★ {score_badge}</p>
                <hr style="opacity:0.1; margin:15px 0;">
                <p style="font-size:0.8rem; color:#888; text-align:left;">
                    The semantic score is calculated using dense vectors generated by a <strong>HuggingFace transformer model (all-MiniLM-L6-v2)</strong>. It measures contextual alignment beyond simple keyword matching.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Skill Match Coverage Card
        st.markdown(
            f"""
            <div class="glass-card score-container" style="margin-top:-10px;">
                <p style="font-size:1.1rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; color:#888;">Technical Skill Match</p>
                <div class="score-number" style="font-size:2.8rem; color:#ffffff;">{res["skills_match_score"]}%</div>
                <p style="font-size:0.85rem; color:#aaa; margin-top:8px;">{len(res["matching_skills"])} of {len(res["matching_skills"]) + len(res["missing_skills"])} skills found</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col_res_2:
        # Skills columns
        st.markdown('<div class="glass-card" style="height: 100%;">', unsafe_allow_html=True)
        st.subheader("Skill Mapping & Gap Analysis")
        
        col_skills_1, col_skills_2 = st.columns(2)
        with col_skills_1:
            st.markdown("##### ✓ Matching Skills (Found)")
            if res["matching_skills"]:
                matched_html = "".join([f'<span class="skill-badge-match">{s}</span>' for s in res["matching_skills"]])
                st.markdown(matched_html, unsafe_allow_html=True)
            else:
                st.info("No matching taxonomy skills identified in your resume.")
                
        with col_skills_2:
            st.markdown("##### ✗ Missing Required Skills")
            if res["missing_skills"]:
                missing_html = "".join([f'<span class="skill-badge-missing">{s}</span>' for s in res["missing_skills"]])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("Fantastic! No critical skills from the job description are missing.")
                
        # Candidate strengths (not requested in JD, but on resume)
        if res["candidate_skills"]:
            st.markdown("<br>##### ⚡ Additional Candidate Skills", unsafe_allow_html=True)
            extra_skills_html = "".join([f'<span class="skill-badge-match" style="background-color:rgba(255,255,255,0.05); color:#ffffff; border-color:rgba(255,255,255,0.15);">{s}</span>' for s in res["candidate_skills"]])
            st.markdown(extra_skills_html, unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

    # Tabs for Generative AI recommendations & Interview Prep
    st.markdown("### Career Assistant & Interview Preparation")
    tab1, tab2, tab3 = st.tabs(["💡 Resume Improvement Suggestions", "📝 Better Wording Suggestions", "🎤 Interview Preparation Kit"])
    
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Actionable Resume Optimization Steps")
        st.markdown(
            "These tips are generated using generative AI based on your missing and matching skills to optimize your resume's impact."
        )
        
        for i, tip in enumerate(res["improvement_tips"], 1):
            st.markdown(f"**{i}.** {tip}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ATS-Friendly Resume Wording Upgrades")
        st.markdown(
            "Use context-rich verbs and metrics instead of vague sentences. Here are dynamic wording templates targeting the missing skills:"
        )
        
        for w in res["wording_suggestions"]:
            if " | " in w:
                parts = w.split(" | ")
                st.markdown(f"🔴 **{parts[0]}**")
                st.markdown(f"🟢 **{parts[1]}**")
            elif "->" in w:
                parts = w.split("->")
                st.markdown(f"🔴 **{parts[0].strip()}**")
                st.markdown(f"🟢 **{parts[1].strip()}**")
            else:
                st.markdown(f"• {w}")
            st.markdown("<hr style='opacity:0.05; margin:10px 0;'>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with tab3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Simulated Interview Preparation Questions")
        st.markdown(
            "Practice these questions to demonstrate capability in both your current skills and target technical requirements."
        )
        
        col_prep_1, col_prep_2, col_prep_3 = st.columns(3)
        
        with col_prep_1:
            st.markdown("##### 💻 Technical Questions")
            for q in res["interview_kit"]["technical"]:
                st.markdown(f"- *{q}*")
                
        with col_prep_2:
            st.markdown("##### 🤝 Behavioral & HR Questions")
            for q in res["interview_kit"]["hr"]:
                st.markdown(f"- *{q}*")
                
        with col_prep_3:
            st.markdown("##### 📐 System & Project Architecture")
            for q in res["interview_kit"]["project"]:
                st.markdown(f"- *{q}*")
                
        st.markdown('</div>', unsafe_allow_html=True)
