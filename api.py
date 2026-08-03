from fastapi import FastAPI
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.workflow import build_workflow
from src.llm import LLMService
import contextlib

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models and initializing graph...")
    app.state.workflow = build_workflow()
    _ = LLMService.get_instance()
    print("Agent is ready to accept queries.")
    yield
    print("Shutting down...")

app = FastAPI(title="OrbitDesk Support Agent API", lifespan=lifespan)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    initial_state = {
        "question": req.question,
        "verification_attempts": 0
    }
    
    final_state = app.state.workflow.invoke(initial_state)
    
    output = {
        "classification": final_state.get("classification"),
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", 0.0),
        "requires_human": final_state.get("requires_human", False),
        "reason": final_state.get("reason", "")
    }
    if final_state.get("clarification_question"):
        output["clarification_question"] = final_state.get("clarification_question")
        
    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
