from typing import TypedDict, List, Dict, Any, Optional

class AgentState(TypedDict):
    question: str
    classification: str  # "answerable", "requires_clarification", "requires_escalation", "out_of_scope", "safe_failure"
    
    # Retrieval
    retrieved_docs: List[Dict[str, str]]
    
    # Generation
    answer: str
    confidence: float
    sources: List[Dict[str, str]]
    requires_human: bool
    reason: str
    clarification_question: Optional[str]
    warnings: List[str]
    
    # Workflow control
    verification_attempts: int
