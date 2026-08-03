from src.state import AgentState
from src.llm import LLMService

def triage_node(state: AgentState) -> dict:
    llm = LLMService.get_instance()
    question = state.get("question", "")
    
    prompt = f"""
You are a support agent triage system. 
Classify the following user question into exactly one of these categories:
- answerable
- requires_clarification
- requires_escalation
- out_of_scope

Examples:
Query: "How do I create an API credential?"
Classification: answerable

Query: "Sync is not working."
Classification: requires_clarification

Query: "Dashboard shows render_failed twice in a row."
Classification: requires_escalation

Query: "Issue a refund for my subscription."
Classification: out_of_scope

Now classify the following query. Reply ONLY with the exact category name. Do not include any other text.
Query: "{question}"
Classification:"""
    messages = [{"role": "user", "content": prompt}]
    result = llm.generate_chat(messages, max_new_tokens=10).strip()
    
    # Clean up result
    result = result.replace('"', '').replace("'", "").strip()
    
    valid_classes = ["answerable", "requires_clarification", "requires_escalation", "out_of_scope"]
    
    classification = "answerable" # fallback
    for cls in valid_classes:
        if cls in result:
            classification = cls
            break
            
    return {"classification": classification, "verification_attempts": 0}
