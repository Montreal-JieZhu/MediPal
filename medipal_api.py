from src import ask

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

class Query(BaseModel):
    query: str

app = FastAPI()

@app.post("/ask")
def medipal_post_api(query: Query):
    user_query = query.query
    ai_response = ask(user_query)
    return {"message": ai_response}

@app.get("/ask")
def medipal_get_api(query: str):
    ai_response = ask(query)
    return {"message": ai_response}
    
def launch_api():
    uvicorn.run(app, host="0.0.0.0", port=30000, log_level="info")
    
__all__ = ["launch_api"]

if __name__ == "__main__":
    launch_api()

# python -m uvicorn medipal_api:app --reload --host 0.0.0.0 --port 30000
