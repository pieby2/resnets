import json
import os
from pathlib import Path
from src.workflow import build_workflow
from src.llm import LLMService

def run_tests():
    base_path = Path(os.path.dirname(os.path.abspath(__file__)))
    questions_file = base_path / "sample_questions.json"
    
    if not questions_file.exists():
        print("sample_questions.json not found.")
        return
        
    with open(questions_file, "r") as f:
        data = json.load(f)
        
    questions = data.get("questions", [])
    
    app = build_workflow()
    
    print("Pre-loading models...")
    # Load LLM to prevent lazy load delay during first query
    _ = LLMService.get_instance()
    
    results = []
    
    for q in questions:
        print(f"\n--- Processing: {q['question_id']} ---")
        print(f"Question: {q['question']}")
        
        initial_state = {
            "question": q["question"],
            "verification_attempts": 0
        }
        
        # Invoke the graph
        final_state = app.invoke(initial_state)
        
        # Format output to match schema
        output = {
            "classification": final_state.get("classification"),
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "confidence": final_state.get("confidence", 0.0),
            "requires_human": final_state.get("requires_human", False),
            "reason": final_state.get("reason", "")
        }
        
        if final_state.get("clarification_question"):
            output["clarification_question"] = final_state["clarification_question"]
            
        print(json.dumps(output, indent=2))
        results.append({
            "question_id": q["question_id"], 
            "question": q["question"], 
            "output": output
        })
        
    # Write to outputs.json
    out_file = base_path / "outputs.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

if __name__ == "__main__":
    run_tests()
