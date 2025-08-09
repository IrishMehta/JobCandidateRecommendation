# Candidate Recommendation Engine

Streamlit app that ranks resumes against a job description using JobBERT‑v2 embeddings and Groq LLM reasoning.

### Features
- Upload multiple resumes (PDF/TXT/DOCX) or paste resume texts
- Parse resumes to text via LlamaParse (markdown output)
- Generate embeddings with `TechWolf/JobBERT-v2` and rank by cosine similarity
- Filter by similarity threshold and top‑K
- Per‑candidate, concise explanation via Groq chat completions
- Optional name extraction from resume headers (spaCy)

### Tech stack
- Streamlit
- sentence-transformers (JobBERT‑v2)
- scikit‑learn (cosine similarity)
- Groq (chat completions; model: `openai/gpt-oss-20b`)
- llama-parse
- pandas, numpy
- python‑dotenv
- spaCy (optional `en_core_web_sm`)

### Setup
1) Python 3.11 recommended
2) Install dependencies:
```
pip install -r requirements.txt
```
3) Create `.env` in project root:
```
LLAMAPARSE=your_llamaparse_api_key
GroqAPI=your_groq_api_key
HFReadToken=your_huggingface_token   
```

### Run
```
streamlit run app.py
```

### Usage
1) Paste the job description
2) Upload files or paste resume texts
3) Adjust similarity threshold and top‑K
4) Click "Find Top Candidates"

### Project structure (compact)
```
CandidateRecommendation/
├── app.py                   # Streamlit UI
├── src/
│   ├── embeddings.py        # JobBERT‑v2 embeddings + ranking
│   ├── resume_parser.py     # LlamaParse extraction
│   ├── reasoning.py         # Groq LLM reasoning
│   └── utils.py             # IO helpers
├── components/
│   ├── file_uploader.py     # Upload/persist resumes
│   ├── results_display.py   # Table + per‑candidate feedback
│   └── sidebar.py           # Threshold + top‑K controls
├── config/
│   └── settings.py          # Defaults (model names, UI labels)
├── data/
│   ├── sample_resumes/
│   ├── temp_uploads/
│   └── embeddings_cache/
└── tests/                   # Basic embedding tests and sample data
```

### Notes
- Environment variables: `LLAMAPARSE`, `GroqAPI`, `HFReadToken`
- Defaults (threshold/top‑K, allowed file types) live in `config/settings.py` 


### Decision Justifications

| Decision | Justification |
|---|---|
| No embedding cache | Resume filenames/sizes often collide; high overwrite risk and low reuse in typical sessions. |
| Use Groq chat completions for feedback | Higher‑quality concise rationales; acceptable cost/latency for a POC. |
| Avoid small summarization models | Token/context limits led to shallow outputs for multi‑page resumes. |
| Minimal name extraction | Assume applicant name is provided; spaCy NER is optional and falls back to filename when missing. |
| Cosine similarity only | Meets requirement with simple, deterministic ranking; avoids reranker complexity. |
| LlamaParse for parsing | Robust across PDF/DOCX with clean markdown output; fewer edge‑case failures than basic PDF libs. |
| Stopword removal before reasoning | Cuts token usage and cost without materially changing meaning. |
| No persistent DB | Store uploads under `data/temp_uploads/` only; simpler setup and easier local runs. |
| JobBERT‑v2 for embeddings | Domain‑specific embeddings improve job‑resume alignment over generic models. |
| Prototype performance trade‑off | ~5–10s/resume (embedding + remote LLM) acceptable for prototype; can be optimized later. |


### Disclaimer
This project was developed with significant assistance from AI tools, with approximately 50% of the codebase being generated or modified by AI. Specifically, the streamlit frontend and the resume parsing are developed completely by AI while other components are created in a pair programming setup. While the code has been reviewed and tested, it's important to note that:

    The system may contain AI-generated code patterns and structures
    Some implementations may follow AI-suggested best practices
    Code quality and security should be thoroughly reviewed before deployment
    Regular updates and maintenance are recommended
