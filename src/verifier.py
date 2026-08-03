from src.state import AgentState
from src.llm import LLMService

def verify_node(state: AgentState) -> dict:
    llm = LLMService.get_instance()
    answer = state.get("answer", "")
    question = state.get("question", "")
    classification = state.get("classification", "answerable")
    
    # We only verify answerable responses
    if classification != "answerable":
        return {"classification": classification}

    docs = state.get("retrieved_docs", [])
    context = "\n".join([f"[{d['source_id']}] {d['passage']}" for d in docs])

    prompt = f"""
You are a verification system. Check if the generated answer is supported by the retrieved evidence.
Answer: "{answer}"
Evidence: "{context}"

Is the answer supported by the evidence? Does it avoid inventing unsupported instructions?
Reply with exactly "PASS" or "FAIL". Do not provide any other explanation.
"""
    messages = [{"role": "user", "content": prompt}]
    result = llm.generate_chat(messages, max_new_tokens=10).strip().upper()
    
    if "FAIL" in result:
        attempts = state.get("verification_attempts", 0) + 1
        if attempts >= 2:
            # Reached max retries, fail safely
            return {"classification": "safe_failure", "verification_attempts": attempts}
        else:
            return {"classification": "needs_revision", "verification_attempts": attempts}
            
    return {"classification": "answerable"}
