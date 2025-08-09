from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from huggingface_hub import login
import os
from dotenv import load_dotenv
from config.settings import EMBEDDING_MODEL
from src.reasoning import generate_comprehensive_fit_reasoning
from src.utils import get_env_var

try:
    from nltk.corpus import stopwords as nltk_stopwords  # type: ignore
except Exception:
    nltk_stopwords = None  # Fallback if NLTK isn't installed or data missing

# A compact fallback English stopword list to avoid hard dependency on NLTK
DEFAULT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without", "to", "from",
    "of", "in", "on", "for", "as", "by", "at", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "into", "over", "under", "about",
    "above", "below", "up", "down", "out", "off", "than", "then", "so", "such", "not", "no",
    "can", "could", "should", "would", "may", "might", "must", "will", "just", "do", "does",
    "did", "doing", "have", "has", "had", "having", "i", "you", "he", "she", "we", "they",
    "me", "him", "her", "us", "them", "my", "your", "his", "her", "our", "their", "mine",
    "yours", "ours", "theirs"
}

load_dotenv()

class JobResumeEmbedder:
    def __init__(self):
        """Initialize the embedding model and optionally login to Hugging Face Hub."""
        hf_token = get_env_var("HFReadToken")
        if hf_token:
            try:
                login(token=hf_token)
            except Exception as e:
                pass
        else:
            pass
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    
    def generate_embedding(self, text):
        """Generate embedding for a single text (job description or resume)."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        return self.model.encode(text, convert_to_tensor=False)
    
    def generate_job_embedding(self, job_description):
        return self.generate_embedding(job_description)
    
    def generate_resume_embedding(self, resume_text):
        return self.generate_embedding(resume_text)
    
    def calculate_similarity(self, job_embedding, resume_embedding):
        job_emb = job_embedding.reshape(1, -1)
        resume_emb = resume_embedding.reshape(1, -1)
        similarity = cosine_similarity(job_emb, resume_emb)[0][0]
        return similarity
    
    def batch_resume_embeddings(self, resume_texts):
        start_time = time.time()
        embeddings = self.model.encode(resume_texts, convert_to_tensor=False, show_progress_bar=True)
        end_time = time.time()
        return embeddings
    
    def rank_candidates(self, job_description, resume_texts, candidate_names=None):
        job_embedding = self.generate_job_embedding(job_description)
        resume_embeddings = self.batch_resume_embeddings(resume_texts)
        similarities = []
        for resume_embedding in resume_embeddings:
            similarity = self.calculate_similarity(job_embedding, resume_embedding)
            similarities.append(similarity)
        if candidate_names is None:
            candidate_names = [f"Candidate_{i+1}" for i in range(len(resume_texts))]
        ranked_candidates = list(zip(candidate_names, similarities, resume_texts))
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        return ranked_candidates
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from the text using NLTK if available, else a small fallback set."""
        if not text:
            return text
        words = text.split()
        # Try NLTK stopwords, fall back to the small built-in list
        stop_set = DEFAULT_STOPWORDS
        if nltk_stopwords is not None:
            try:
                stop_set = set(nltk_stopwords.words('english'))
            except Exception:
                # Missing corpus; keep fallback
                pass
        filtered = [word for word in words if word.lower() not in stop_set]
        return " ".join(filtered)

    def generate_fit_reasoning(self, job_description, resume_text, candidate_name):
        """Generate reasoning using the dedicated reasoning module."""
        # to reduce the token usage, we can remove the stopwords from the job description and resume text
        job_description = self.remove_stopwords(job_description)
        resume_text = self.remove_stopwords(resume_text)
        return generate_comprehensive_fit_reasoning(job_description, resume_text, candidate_name)

    def summarize_top_candidates(self, job_description, ranked_candidates, top_k=10, return_markdown=True):
        top = ranked_candidates[:top_k]
        rows = []

        for name, score, resume_text in top:
            reasoning = self.generate_fit_reasoning(
                job_description, resume_text, name
            )
            rows.append({
                "name": name,
                "similarity": float(score),
                "reasoning": reasoning,
            })

        if not return_markdown:
            return rows
        header = "| Name | Similarity Score | Reasoning |\n|---|---:|---|"
        body_lines = [f"| {r['name']} | {r['similarity']:.4f} | {r['reasoning']} |" for r in rows]
        table_md = "\n".join([header] + body_lines)
        return table_md
