from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os, json, time
from openai import OpenAI

app = FastAPI(
    title="RankIQ — AI Candidate Ranking",
    description="LLM + Vector Search powered candidate ranking system by Team VLXTech",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Models ──────────────────────────────────────────────────────────────────

class CandidateProfile(BaseModel):
    candidate_id: str
    name: str
    skills: list[str]
    years_of_experience: int
    domain: str
    education: str
    projects: Optional[str] = None
    profile_text: str

class RankRequest(BaseModel):
    job_description: str
    candidates: list[CandidateProfile]
    top_k: int = 10

class CandidateScore(BaseModel):
    candidate_id: str
    name: str
    score: float
    rank: int
    justification: str
    signals: dict
    confidence_flag: Optional[str] = None

class RankResponse(BaseModel):
    rankings: list[CandidateScore]
    processing_time_ms: float

# ── Core Logic ───────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def detect_suspicious_profile(candidate: CandidateProfile) -> Optional[str]:
    if candidate.years_of_experience > 40:
        return "Inflated years of experience"
    skills_text = " ".join(candidate.skills).lower()
    keyword_spam = len(candidate.skills) > 30
    if keyword_spam:
        return "Possible keyword stuffing detected"
    return None

def rerank_with_llm(jd: str, candidate: CandidateProfile, vector_score: float) -> dict:
    prompt = f"""You are an expert technical recruiter. Score this candidate against the job description.

JOB DESCRIPTION:
{jd}

CANDIDATE PROFILE:
Name: {candidate.name}
Skills: {', '.join(candidate.skills)}
Experience: {candidate.years_of_experience} years
Domain: {candidate.domain}
Education: {candidate.education}
Projects: {candidate.projects or 'N/A'}
Profile: {candidate.profile_text}

Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "skill_match": <0-100>,
  "experience_match": <0-100>,
  "domain_fit": <0-100>,
  "profile_quality": <0-100>,
  "justification": "<one sentence explaining the ranking>"
}}"""

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=300
    )
    raw = resp.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "RankIQ API is running 🚀", "docs": "/docs"}

@app.post("/rank", response_model=RankResponse)
def rank_candidates(req: RankRequest):
    start = time.time()
    jd_embedding = get_embedding(req.job_description)
    results = []

    for candidate in req.candidates:
        # Vector similarity
        cand_text = f"{candidate.profile_text} {' '.join(candidate.skills)} {candidate.domain}"
        cand_embedding = get_embedding(cand_text)
        vector_score = cosine_similarity(jd_embedding, cand_embedding) * 100

        # LLM re-ranking
        try:
            llm_scores = rerank_with_llm(req.job_description, candidate, vector_score)
        except Exception:
            llm_scores = {
                "skill_match": vector_score,
                "experience_match": 50,
                "domain_fit": 50,
                "profile_quality": 50,
                "justification": "Scored via semantic similarity."
            }

        # Weighted composite score
        composite = (
            llm_scores["skill_match"] * 0.40 +
            llm_scores["experience_match"] * 0.30 +
            llm_scores["domain_fit"] * 0.20 +
            llm_scores["profile_quality"] * 0.10
        )

        flag = detect_suspicious_profile(candidate)

        results.append({
            "candidate_id": candidate.candidate_id,
            "name": candidate.name,
            "score": round(composite, 2),
            "justification": llm_scores.get("justification", ""),
            "signals": {
                "skill_match": llm_scores["skill_match"],
                "experience_match": llm_scores["experience_match"],
                "domain_fit": llm_scores["domain_fit"],
                "profile_quality": llm_scores["profile_quality"],
                "vector_similarity": round(vector_score, 2)
            },
            "confidence_flag": flag
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:req.top_k]
    for i, r in enumerate(top):
        r["rank"] = i + 1

    elapsed = (time.time() - start) * 1000
    return RankResponse(
        rankings=[CandidateScore(**r) for r in top],
        processing_time_ms=round(elapsed, 1)
    )

@app.get("/health")
def health():
    return {"status": "ok"}
