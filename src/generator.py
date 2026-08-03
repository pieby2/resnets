from src.state import AgentState
from src.llm import LLMService

def generate_node(state: AgentState) -> dict:
    llm = LLMService.get_instance()
    question = state.get("question", "")
    classification = state.get("classification", "answerable")
    
    if classification == "requires_clarification":
        return {
            "answer": "I need more information to answer your question.",
            "requires_human": False,
            "clarification_question": "Could you please provide more details?",
            "confidence": 1.0,
            "reason": "Question is ambiguous.",
            "sources": []
        }
    elif classification == "requires_escalation":
        return {
            "answer": "I need to escalate this issue to our support team.",
            "requires_human": True,
            "confidence": 1.0,
            "reason": "Requires human intervention/engineering team.",
            "sources": []
        }
    elif classification == "out_of_scope":
        return {
            "answer": "I cannot help with this request as it is outside the scope of this support system.",
            "requires_human": False,
            "confidence": 1.0,
            "reason": "Out of scope request.",
            "sources": []
        }
    elif classification == "safe_failure":
        return {
            "answer": "I am unable to verify my answer based on the available documentation. Please contact a human agent.",
            "requires_human": True,
            "confidence": 0.0,
            "reason": "Failed verification steps safely.",
            "sources": []
        }
    
    # Otherwise, it's answerable. We generate a response based on evidence.
    docs = state.get("retrieved_docs", [])
    context_parts = []
    sources = []
    for d in docs:
        context_parts.append(f"Source ID: {d['source_id']}\nPassage: {d['passage']}")
        sources.append({"source_id": d["source_id"], "passage": d["passage"]})
        
    context = "\n\n".join(context_parts)
    
    prompt = f"""
You are a helpful support agent for the product OrbitDesk.
Answer the user's question based ONLY on the provided evidence. 
Do not invent unsupported instructions.
If the evidence does not contain the answer, say "I cannot answer this based on the available information."

Evidence:
{context}

Question: {question}

Provide only the final answer text. Do not include any meta commentary.
"""
    messages = [{"role": "user", "content": prompt}]
    result = llm.generate_chat(messages, max_new_tokens=300).strip()
    
    # Attempting to assign a pseudo-confidence
    confidence = 0.9 if len(docs) > 0 else 0.2
    
    return {
        "answer": result,
        "sources": sources,
        "confidence": confidence,
        "requires_human": False,
        "reason": "Answered from retrieved context."
    }
