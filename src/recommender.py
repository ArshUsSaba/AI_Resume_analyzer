import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from src.utils import log_info, log_error, get_gpu_status

class CareerRecommender:
    def __init__(self, model_name='google/flan-t5-base', use_llm=True):
        """
        Initializes the Generative AI recommendation engine.
        Args:
            model_name (str): Hugging Face model identifier for FLAN-T5.
            use_llm (bool): If True, attempts to load the HF transformer model. 
                            If False or if loading fails, falls back to rule-based generation.
        """
        self.use_llm = use_llm
        self.model_name = model_name
        self.pipeline = None
        
        if self.use_llm:
            try:
                gpu_status = get_gpu_status()
                device = 0 if gpu_status["available"] else -1
                device_str = "GPU (cuda)" if device == 0 else "CPU"
                
                log_info(f"Loading Generative AI model '{model_name}' on {device_str}...")
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # To prevent memory issues, we load in float16 if GPU is available
                if device == 0:
                    model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                
                self.pipeline = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device_map="auto" if device == 0 else None,
                    device=device if device == -1 else None
                )
                log_info(f"Generative AI model loaded successfully.")
            except Exception as e:
                log_error(f"Failed to load LLM pipeline for '{model_name}'. Falling back to rule-based recommendations.", e)
                self.use_llm = False

    def generate_improvement_tips(self, missing_skills, matching_skills):
        """
        Generates resume improvement suggestions.
        """
        if not missing_skills:
            return ["Excellent! Your resume already matches all key skills identified in the job description. Focus on detailing project metrics and leadership roles."]

        if self.use_llm and self.pipeline:
            prompt = (
                f"Resume improvements. The candidate is missing these skills: {', '.join(missing_skills)}. "
                f"They have these matching skills: {', '.join(matching_skills)}. "
                f"Write a bulleted list of 3 specific, actionable resume improvement suggestions to address these missing skills."
            )
            try:
                result = self.pipeline(prompt, max_length=256, num_return_sequences=1, temperature=0.7, do_sample=True)
                generated_text = result[0]['generated_text']
                # Split and clean bullet points
                tips = [tip.strip().lstrip('-*•').strip() for tip in generated_text.split('\n') if tip.strip()]
                if tips:
                    return tips
            except Exception as e:
                log_error("LLM improvement tips generation failed. Using fallback.", e)
        
        # Rule-based fallback
        tips = []
        for i, skill in enumerate(missing_skills[:3]):
            if skill in ["CUDA", "TensorRT", "GPU Acceleration", "Parallel Computing"]:
                tips.append(f"Add a dedicated project demonstrating optimization of model inference using {skill} to showcase performance engineering capabilities.")
            elif skill in ["PyTorch", "TensorFlow", "Keras", "Scikit-Learn"]:
                tips.append(f"Incorporate a section highlighting end-to-end model development cycle (data pipeline, training, validation) built with {skill}.")
            elif skill in ["Docker", "Kubernetes", "CI/CD"]:
                tips.append(f"Describe how you containerized your ML models using {skill} to build reproducible, cloud-ready deployment environments.")
            elif skill in ["AWS", "GCP", "Azure"]:
                tips.append(f"Quantify your cloud deployments. E.g., 'Deployed deep learning models on {skill} using Serverless/Kubernetes, reducing API latency by 20%'.")
            else:
                tips.append(f"Acquire foundation certification or complete a capstone project utilizing {skill} and list it explicitly under your projects or skills section.")
        
        # If no specific skill tips matched, provide generic tips
        if not tips:
            tips = [
                "Quantify your achievements using the XYZ framework (e.g. 'Accomplished [X] as measured by [Y], by doing [Z]').",
                "Move skills mentioned in descriptions into a structured 'Technical Skills' section at the top for ATS parsing.",
                "Structure your projects list by clearly highlighting: Problem, Action (Tech Stack), and Business/Technical Impact."
            ]
        return tips

    def generate_wording_suggestions(self, missing_skills):
        """
        Generates better technical wording suggestions for the resume.
        """
        if not missing_skills:
            return ["No wording modifications needed. Focus on emphasizing scale, throughput, and business metrics."]

        if self.use_llm and self.pipeline:
            prompt = (
                f"Wording suggestions. Give 3 examples of how to rewrite bullet points in a resume to sound more professional "
                f"and include the following technologies: {', '.join(missing_skills[:3])}. "
                f"Format as: 'Before: ... | After: ...'"
            )
            try:
                result = self.pipeline(prompt, max_length=256, num_return_sequences=1, temperature=0.7, do_sample=True)
                generated_text = result[0]['generated_text']
                wording = [w.strip() for w in generated_text.split('\n') if w.strip()]
                if wording:
                    return wording
            except Exception as e:
                log_error("LLM wording suggestions generation failed. Using fallback.", e)
        
        # Rule-based fallback
        wording_map = {
            "CUDA": "Before: 'Wrote code for GPU speedups.' -> After: 'Designed and optimized parallelized custom operations in C++/CUDA, resulting in a 5x acceleration of tensor operations.'",
            "PyTorch": "Before: 'Used PyTorch for classification.' -> After: 'Architected and trained deep neural networks using PyTorch's autograd and DistributedDataParallel, enhancing model accuracy by 14%.'",
            "Docker": "Before: 'Set up Docker containers.' -> After: 'Streamlined delivery pipeline by containerizing microservices with Docker, reducing developer onboarding time by 40%.'",
            "Kubernetes": "Before: 'Deployed application on server.' -> After: 'Orchestrated highly available ML microservices using Kubernetes (K8s), ensuring 99.9% uptime and automatic horizontal scaling.'",
            "Machine Learning": "Before: 'Built ML models.' -> After: 'Developed and deployed predictive machine learning models using Scikit-Learn, yielding an annual cost reduction of $50k.'"
        }
        
        suggestions = []
        for skill in missing_skills:
            if skill in wording_map:
                suggestions.append(wording_map[skill])
            if len(suggestions) >= 3:
                break
                
        if len(suggestions) < 3:
            suggestions.append("Before: 'Responsible for cleaning data and creating charts.' -> After: 'Engineered robust ETL pipelines processing over 10M rows daily, utilizing Pandas and SQL for feature extraction.'")
            suggestions.append("Before: 'Helped speed up the website database.' -> After: 'Optimized database queries and structured indexing in PostgreSQL, reducing query latency by 45%.'")
            
        return suggestions[:3]

    def generate_interview_questions(self, matching_skills, missing_skills, job_title="AI Engineer"):
        """
        Generates interview questions based on the candidate's skills and the target role.
        """
        if self.use_llm and self.pipeline:
            prompt = (
                f"Generate 2 technical interview questions about {', '.join(matching_skills[:2]) or 'Machine Learning'}, "
                f"1 technical question about a missing skill {', '.join(missing_skills[:1]) or 'System Design'}, "
                f"1 HR question about situational leadership, and 1 project architecture question for a {job_title} role."
            )
            try:
                result = self.pipeline(prompt, max_length=256, num_return_sequences=1, temperature=0.7, do_sample=True)
                generated_text = result[0]['generated_text']
                questions = [q.strip().lstrip('123456789.-*• ').strip() for q in generated_text.split('\n') if q.strip()]
                
                # Standardize output structure
                if len(questions) >= 3:
                    return {
                        "technical": questions[:3],
                        "hr": [questions[3]] if len(questions) > 3 else ["Tell me about a time you resolved a major bug under a tight deadline."],
                        "project": [questions[4]] if len(questions) > 4 else ["Can you explain the system architecture of your most complex project, including data flow?"]
                    }
            except Exception as e:
                log_error("LLM interview question generation failed. Using fallback.", e)
                
        # Rule-based fallback
        tech_questions = []
        if "CUDA" in missing_skills or "CUDA" in matching_skills:
            tech_questions.append("Explain the memory hierarchy in CUDA (global, shared, registers). How do you optimize shared memory access to avoid bank conflicts?")
        if "PyTorch" in missing_skills or "PyTorch" in matching_skills:
            tech_questions.append("What is the difference between PyTorch's `Dataset` and `DataLoader`? Explain how you configure multi-process data loading.")
        if "Machine Learning" in matching_skills or "Deep Learning" in matching_skills:
            tech_questions.append("How do you handle exploding or vanishing gradients during training of deep neural networks? Explain gradient clipping.")
        
        # Add general tech questions if we need more
        while len(tech_questions) < 3:
            tech_questions.append("What is the difference between L1 and L2 regularization? How do they affect model weights mathematically?")
            tech_questions.append("Describe the concept of 'attention' in Transformer architectures. How does self-attention scale with sequence length?")
            
        hr_questions = [
            "Tell me about a time you had to learn a complex new framework or technology (like CUDA or PyTorch) quickly for a project. How did you approach it?",
            "Describe a situation where you had a disagreement with a team member about a technical design decision. How did you resolve it?"
        ]
        
        project_questions = [
            "Explain the database schema and data flow for a project on your resume. Why did you choose that specific design over alternatives?",
            "How did you validate your model's offline performance, and what metrics did you use to track its success in production?"
        ]
        
        return {
            "technical": tech_questions[:3],
            "hr": hr_questions[:2],
            "project": project_questions[:2]
        }
