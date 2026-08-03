from typing import Literal
from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.retriever import DocumentRetriever
from src.triage import triage_node
from src.generator import generate_node
from src.verifier import verify_node
import os

retriever = None

def init_retriever():
    global retriever
    if retriever is None:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        retriever = DocumentRetriever(base_path)

def retrieve_node(state: AgentState) -> dict:
    global retriever
    if retriever is None:
        init_retriever()
    
    question = state.get("question", "")
    docs = retriever.retrieve(question)
    return {"retrieved_docs": docs}

def route_triage(state: AgentState):
    cls = state.get("classification")
    if cls == "answerable":
        return "retrieve"
    else:
        return "generate"

def route_verify(state: AgentState):
    cls = state.get("classification")
    if cls == "needs_revision" or cls == "safe_failure":
        return "generate"
    return END

def build_workflow():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("verify", verify_node)
    
    # Add edges
    workflow.add_edge(START, "triage")
    workflow.add_conditional_edges(
        "triage", 
        route_triage, 
        {
            "retrieve": "retrieve",
            "generate": "generate"
        }
    )
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "verify")
    workflow.add_conditional_edges(
        "verify", 
        route_verify, 
        {
            "generate": "generate",
            END: END
        }
    )
    
    app = workflow.compile()
    return app
