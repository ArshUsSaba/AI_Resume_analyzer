import time
from src.pdf_reader import extract_text_from_pdf
from src.embeddings import ResumeMatcher
from src.skill_analyzer import analyze_skills
from src.recommender import CareerRecommender

def main():
    print("=== Testing PDF Extraction ===")
    resume_text = extract_text_from_pdf("sample_data/sample_resume.pdf")
    print(f"Extracted {len(resume_text)} characters.")
    
    with open("sample_data/sample_job_description.txt", "r", encoding="utf-8") as f:
        job_desc = f.read()
    
    print("\n=== Testing Embeddings & Similarity ===")
    matcher = ResumeMatcher()
    similarity = matcher.compute_similarity(resume_text, job_desc)
    print(f"Semantic Similarity: {similarity}%")
    
    print("\n=== Testing Skill Analysis ===")
    skills_results = analyze_skills(resume_text, job_desc)
    print(f"Matching Skills: {skills_results['matching_skills']}")
    print(f"Missing Skills: {skills_results['missing_skills']}")
    print(f"Skills Match Score: {skills_results['skills_match_score']}%")
    
    print("\n=== Testing Recommendations (Rule-Based Fallback) ===")
    # Initialize recommender with use_llm=False to run instantly
    recommender = CareerRecommender(use_llm=False)
    tips = recommender.generate_improvement_tips(skills_results['missing_skills'], skills_results['matching_skills'])
    wording = recommender.generate_wording_suggestions(skills_results['missing_skills'])
    questions = recommender.generate_interview_questions(skills_results['matching_skills'], skills_results['missing_skills'])
    
    print("Tips:")
    for t in tips:
        print(f" - {t}")
    print("Wording suggestions:")
    for w in wording:
        print(f" - {w}")
    print("Interview Questions:")
    for cat, qs in questions.items():
        print(f" {cat.upper()}:")
        for q in qs:
            print(f"   * {q}")

if __name__ == "__main__":
    main()
