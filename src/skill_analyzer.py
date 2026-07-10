import re
from src.utils import log_info

# Define a comprehensive skills taxonomy relevant to Software Eng, AI/ML, DevOps, and Data Science
SKILL_TAXONOMY = {
    # Programming Languages
    "Python": [r"\bpython\b"],
    "C++": [r"\bc\+\+\b", r"\bcpp\b"],
    "C#": [r"\bc#\b", r"\bc-sharp\b"],
    "Java": [r"\bjava\b"],
    "JavaScript": [r"\bjavascript\b", r"\bjs\b"],
    "TypeScript": [r"\btypescript\b", r"\bts\b"],
    "Go": [r"\bgo\b", r"\bgolang\b"],
    "Rust": [r"\brust\b"],
    "SQL": [r"\bsql\b", r"\bmysql\b", r"\bpostgresql\b", r"\bsqlite\b"],
    "R Language": [r"\b(r\s*programming|r-project|r-lang)\b"],
    "HTML/CSS": [r"\bhtml5?\b", r"\bcss3?\b"],
    "Bash": [r"\bbash\b", r"\bshell\s+scripting\b"],
    "Scala": [r"\bscala\b"],
    "C Language": [r"\b(c\s*programming)\b", r"\bansi\s+c\b"],
    
    # Machine Learning & AI
    "Machine Learning": [r"\bmachine\s+learning\b", r"\bml\b"],
    "Deep Learning": [r"\bdeep\s+learning\b", r"\bdl\b"],
    "Neural Networks": [r"\bneural\s+networks?\b", r"\banns?\b", r"\bcnns?\b", r"\brnns?\b"],
    "Natural Language Processing": [r"\bnatural\s+language\s+processing\b", r"\bnlp\b"],
    "Computer Vision": [r"\bcomputer\s+vision\b", r"\bcv\b"],
    "Generative AI": [r"\bgenerative\s+ai\b", r"\bgenai\b"],
    "Large Language Models": [r"\blarge\s+language\s+models?\b", r"\bllms?\b"],
    "Transformers": [r"\btransformers?\b", r"\battention\s+mechanism\b"],
    "Reinforcement Learning": [r"\breinforcement\s+learning\b", r"\brl\b"],
    "Prompt Engineering": [r"\bprompt\s+engineering\b"],
    "RAG": [r"\brag\b", r"\bretrieval\s+augmented\s+generation\b"],
    
    # AI/ML Libraries & Frameworks
    "PyTorch": [r"\bpy\s*torch\b"],
    "TensorFlow": [r"\btensor\s*flow\b"],
    "Keras": [r"\bkeras\b"],
    "Scikit-Learn": [r"\bscikit-learn\b", r"\bsci-kit\s*learn\b", r"\bsklearn\b"],
    "Pandas": [r"\bpandas\b"],
    "NumPy": [r"\bnumpy\b"],
    "Hugging Face": [r"\bhugging\s*face\b", r"\bhf\b"],
    "OpenCV": [r"\bopencv\b"],
    "SpaCy": [r"\bspacy\b"],
    "NLTK": [r"\bnltk\b"],
    
    # Hardware & Performance (NVIDIA Specific & High Performance)
    "CUDA": [r"\bcuda\b", r"\bcompute\s+unified\s+device\s+architecture\b"],
    "TensorRT": [r"\btensorrt\b", r"\btensor-rt\b"],
    "ONNX": [r"\bonnx\b", r"\bopen\s+neural\s+network\s+exchange\b"],
    "OpenMP": [r"\bopenmp\b"],
    "MPI": [r"\bmpi\b", r"\bmessage\s+passing\s+interface\b"],
    "GPU Acceleration": [r"\bgpus?\b", r"\bgpu\s+acceleration\b", r"\bgpu\s+computing\b"],
    "Parallel Computing": [r"\bparallel\s+computing\b", r"\bparallel\s+programming\b", r"\bmultithreading\b"],
    "Quantization": [r"\bquantization\b", r"\bmodel\s+quantization\b", r"\bprecision\s+calibration\b"],
    
    # Cloud, DevOps & Data
    "Docker": [r"\bdocker\b", r"\bdocker\s+containers\b"],
    "Kubernetes": [r"\bkubernetes\b", r"\bk8s\b"],
    "AWS": [r"\baws\b", r"\bamazon\s+web\s+services\b"],
    "GCP": [r"\bgcp\b", r"\bgoogle\s+cloud\b", r"\bgoogle\s+cloud\s+platform\b"],
    "Azure": [r"\bazure\b", r"\bmicrosoft\s+azure\b"],
    "Git": [r"\bgit\b", r"\bgithub\b", r"\bgitlab\b"],
    "CI/CD": [r"\bci/cd\b", r"\bcontinuous\s+integration\b"],
    "Linux": [r"\blinux\b", r"\bubuntu\b", r"\bdebian\b", r"\bredhat\b"],
    "Apache Spark": [r"\bspark\b", r"\bapache\s+spark\b", r"\bpyspark\b"],
    "Apache Kafka": [r"\bkafka\b", r"\bapache\s+kafka\b"],
    "ETL": [r"\betl\b", r"\bdata\s+pipeline\b", r"\bdata\s+pipelines\b"],
    
    # Software Engineering & Methodology
    "Agile": [r"\bagile\b", r"\bagile\s+methodology\b"],
    "Scrum": [r"\bscrum\b"],
    "Microservices": [r"\bmicroservices\b", r"\bmicro-services\b"],
    "REST APIs": [r"\brest\s+apis?\b", r"\brestful\b", r"\brest\s+web\s+services\b"],
    "System Design": [r"\bsystem\s+design\b", r"\bsystem\s+architecture\b"]
}

def extract_skills_from_text(text):
    """
    Scans a text document against the skills taxonomy and returns a set of found skills.
    
    Args:
        text (str): Input text (resume or job description).
        
    Returns:
        set: A set of standardized skill names that match the text.
    """
    found_skills = set()
    if not text:
        return found_skills
        
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    for skill_name, patterns in SKILL_TAXONOMY.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_skills.add(skill_name)
                break # Move to next skill once a pattern matches
                
    return found_skills

def analyze_skills(resume_text, job_desc_text):
    """
    Extracts skills from both resume and job description, then identifies
    matching, missing, and candidate-only skills.
    
    Args:
        resume_text (str): Resume text content.
        job_desc_text (str): Job description text content.
        
    Returns:
        dict: Containing 'matching_skills', 'missing_skills', 'candidate_skills', and 'skills_match_score'
    """
    resume_skills = extract_skills_from_text(resume_text)
    job_skills = extract_skills_from_text(job_desc_text)
    
    log_info(f"Extracted {len(resume_skills)} skills from Resume: {resume_skills}")
    log_info(f"Extracted {len(job_skills)} skills from Job Description: {job_skills}")
    
    matching_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills.difference(resume_skills)
    candidate_skills = resume_skills.difference(job_skills)
    
    # Calculate a simple skill coverage score (0-100)
    # If the job description requires no skills in our taxonomy, return 100
    if len(job_skills) == 0:
        skills_match_score = 100.0
    else:
        skills_match_score = round((len(matching_skills) / len(job_skills)) * 100, 2)
        
    return {
        "resume_skills": sorted(list(resume_skills)),
        "job_skills": sorted(list(job_skills)),
        "matching_skills": sorted(list(matching_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "candidate_skills": sorted(list(candidate_skills)),
        "skills_match_score": skills_match_score
    }
