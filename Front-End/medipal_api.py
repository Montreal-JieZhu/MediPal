from fastapi import FastAPI
from pydantic import BaseModel
from medipal import ask
# def ask(query:str)->str:
#     return "hi, how are you!"
# Define request body format
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

# python -m uvicorn medipal_api:app --reload --host 0.0.0.0 --port 30000
