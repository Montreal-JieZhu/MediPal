from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}



# python -m uvicorn test_fastapi:app --reload --host 0.0.0.0 --port 8080
