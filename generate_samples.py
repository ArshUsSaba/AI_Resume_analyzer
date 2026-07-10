import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_resume_pdf(filename):
    """
    Generates a professional resume PDF with pre-determined skills using reportlab.
    """
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Title / Header
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 50, "John Doe")
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 68, "San Francisco, CA | john.doe@email.com | (123) 456-7890 | github.com/johndoe")
    
    # Summary Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 95, "PROFESSIONAL SUMMARY")
    c.setLineWidth(0.5)
    c.line(50, height - 98, width - 50, height - 98)
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 112, "Aspiring Machine Learning Engineer with 2+ years of experience building and deploying predictive models.")
    c.drawString(50, height - 126, "Proficient in Python, machine learning, and deep neural networks using TensorFlow. Passionate about data processing,")
    c.drawString(50, height - 140, "statistical analysis, and optimizing model training pipelines. Strong collaborative skill set under Agile frameworks.")
    
    # Experience Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 165, "WORK EXPERIENCE")
    c.line(50, height - 168, width - 50, height - 168)
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 182, "Junior ML Developer | TechSolutions Inc.")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(450, height - 182, "June 2024 - Present")
    
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 196, "- Built and deployed machine learning models using Scikit-Learn and Pandas for customer churn prediction.")
    c.drawString(60, height - 210, "- Designed deep neural networks using TensorFlow and Keras to classify image datasets, improving accuracy by 10%.")
    c.drawString(60, height - 224, "- Engineered data ETL pipelines in SQL and Python to process 500k records daily, speeding up reporting by 25%.")
    c.drawString(60, height - 238, "- Utilized Git and GitHub for version control and collaborative development within an Agile/Scrum team.")
    
    # Projects Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 265, "ACADEMIC & PERSONAL PROJECTS")
    c.line(50, height - 268, width - 50, height - 268)
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 282, "AI Semantic Search Engine")
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 296, "- Developed a semantic search application using Hugging Face transformers and Python.")
    c.drawString(60, height - 310, "- Evaluated text representations using cosine similarity to retrieve relevant text passages based on user queries.")
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 330, "Automated Document Classifier")
    c.setFont("Helvetica", 10)
    c.drawString(60, height - 344, "- Implemented natural language processing (NLP) tokenization and text analysis to filter spam text messages.")
    c.drawString(60, height - 358, "- Utilized Flask to build web APIs, serving model inferences with sub-100ms response times.")
    
    # Education Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 385, "EDUCATION")
    c.line(50, height - 388, width - 50, height - 388)
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, height - 402, "B.S. in Computer Science | State University")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(450, height - 402, "Graduated: May 2024")
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 416, "GPA: 3.8/4.0 | Focus: Software Engineering & Data Analysis")
    
    # Skills Section
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 442, "TECHNICAL SKILLS")
    c.line(50, height - 445, width - 50, height - 445)
    
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 458, "Programming Languages: Python, SQL, JavaScript, HTML, CSS, Bash")
    c.drawString(50, height - 472, "Libraries & Frameworks: TensorFlow, Keras, Scikit-Learn, Pandas, NumPy, Hugging Face, Flask")
    c.drawString(50, height - 486, "Tools & Methodologies: Git, Agile, Scrum, ETL, REST APIs")
    
    c.save()
    print(f"Generated PDF resume at: {filename}")

def create_job_description_txt(filename):
    """
    Generates a sample job description text file.
    """
    jd_content = """Position: Machine Learning Engineer - AI Platform (NVIDIA Capstone Target)
Company: NVIDIA Corporation
Location: Santa Clara, CA (Hybrid)

About the Team:
Our team builds hardware-accelerated deep learning platforms and runtime engines that power the world's most advanced AI models. We bridge the gap between high-level ML frameworks and specialized GPU architectures.

Key Responsibilities:
- Develop, optimize, and deploy deep learning models using PyTorch, TensorFlow, and custom operations.
- Leverage GPU acceleration using CUDA programming and parallel computing primitives.
- Optimize deep learning inference performance using NVIDIA TensorRT, ONNX, and model quantization.
- Containerize machine learning workloads with Docker and deploy using Kubernetes (k8s) clusters.
- Establish robust automated CI/CD pipelines to build, package, and release model serving endpoints.
- Design efficient ETL data pipelines to process massive multimodal datasets.
- Implement system designs that ensure high availability, parallel computing throughput, and low latency.

Basic Qualifications:
- Bachelor's or Master's degree in Computer Science, Computer Engineering, or related technical discipline.
- Solid software engineering foundations in Python, C++, and Bash scripting.
- Practical experience with deep learning models, particularly Transformers and neural networks.
- Basic understanding of CUDA and GPU programming paradigms.
- Experience with DevOps tools: Git, Docker, Kubernetes, and CI/CD pipelines.
- Strong understanding of relational databases and SQL queries.
- Familiarity with Agile software development practices.
"""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(jd_content)
    print(f"Generated Job Description TXT at: {filename}")

if __name__ == "__main__":
    os.makedirs("C:/Users/user/.gemini/antigravity/scratch/AI-Resume-Analyzer/sample_data", exist_ok=True)
    create_resume_pdf("C:/Users/user/.gemini/antigravity/scratch/AI-Resume-Analyzer/sample_data/sample_resume.pdf")
    create_job_description_txt("C:/Users/user/.gemini/antigravity/scratch/AI-Resume-Analyzer/sample_data/sample_job_description.txt")
