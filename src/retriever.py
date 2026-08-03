import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss

class DocumentRetriever:
    def __init__(self, base_path: str, model_name="all-MiniLM-L6-v2", reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.base_path = Path(base_path)
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Loading re-ranking model: {reranker_name}")
        self.reranker = CrossEncoder(reranker_name)
        self.documents = []  # list of {"source_id": str, "passage": str}
        self.index = None
        self._load_documents()
        self._build_index()

    def _load_documents(self):
        # Load KB
        kb_path = self.base_path / "knowledge_base"
        if kb_path.exists():
            for md_file in kb_path.glob("*.md"):
                content = md_file.read_text(encoding="utf-8")
                # Split by double newline to get rough paragraphs/sections
                paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
                for p in paragraphs:
                    if len(p) > 20: # ignore very short artifacts
                        self.documents.append({
                            "source_id": md_file.name,
                            "passage": p
                        })
        
        # Load resolved cases
        cases_file = self.base_path / "resolved_cases.json"
        if cases_file.exists():
            with open(cases_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                cases = data.get("cases", [])
                for case in cases:
                    if case.get("status") != "superseded":
                        title = case.get("title", "")
                        symptoms = " ".join(case.get("symptoms", []))
                        resolution = " ".join(case.get("resolution", []))
                        passage = f"Case: {title}. Symptoms: {symptoms}. Resolution: {resolution}."
                        if "important_limit" in case:
                            passage += f" Important Limit: {case['important_limit']}"
                        
                        self.documents.append({
                            "source_id": case.get("case_id"),
                            "passage": passage
                        })

    def _build_index(self):
        if not self.documents:
            return
        passages = [doc["passage"] for doc in self.documents]
        embeddings = self.model.encode(passages, convert_to_numpy=True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 3, fetch_k: int = 10):
        if not self.index:
            return []
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, fetch_k)
        
        candidates = []
        for idx in indices[0]:
            if idx < len(self.documents):
                candidates.append(self.documents[idx])
                
        if not candidates:
            return []
            
        # Cross-Encoder Re-ranking
        cross_inp = [[query, doc["passage"]] for doc in candidates]
        cross_scores = self.reranker.predict(cross_inp)
        
        # Sort candidates by score descending
        scored_candidates = sorted(zip(cross_scores, candidates), key=lambda x: x[0], reverse=True)
        
        results = [doc for score, doc in scored_candidates[:top_k]]
        return results
