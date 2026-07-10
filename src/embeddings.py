import torch
from sentence_transformers import SentenceTransformer, util
from src.utils import log_info, get_gpu_status

class ResumeMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initializes the sentence transformer model on GPU if available.
        """
        gpu_status = get_gpu_status()
        self.device = "cuda" if gpu_status["available"] else "cpu"
        log_info(f"Initializing SentenceTransformer model '{model_name}' on device: {self.device}")
        
        # Load the model onto the target device
        self.model = SentenceTransformer(model_name, device=self.device)
        log_info("SentenceTransformer model loaded successfully.")

    def compute_similarity(self, resume_text, job_desc_text):
        """
        Computes the semantic cosine similarity score between the resume and job description.
        
        Args:
            resume_text (str): Extracted resume text.
            job_desc_text (str): Paste job description.
            
        Returns:
            float: Similarity match percentage (0 to 100).
        """
        if not resume_text.strip() or not job_desc_text.strip():
            return 0.0

        # Encode both texts to retrieve embeddings
        resume_emb = self.model.encode(resume_text, convert_to_tensor=True, show_progress_bar=False)
        job_emb = self.model.encode(job_desc_text, convert_to_tensor=True, show_progress_bar=False)

        # Compute cosine similarity
        cosine_score = util.cos_sim(resume_emb, job_emb)
        
        # Get float value and convert to percentage format
        score = float(cosine_score[0][0])
        match_percentage = round(score * 100, 2)
        
        # Clip values to range [0, 100] just in case of float tolerances
        match_percentage = max(0.0, min(100.0, match_percentage))
        
        log_info(f"Computed semantic match score: {match_percentage}%")
        return match_percentage
