from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

# Initialize FastAPI
app = FastAPI()

# Set your OpenAI API-K
openai_api = ""
client = OpenAI(api_key=openai_api)

# Request Model
class QueryRequest(BaseModel):
    message: str

@app.post("/generate_prompt")
async def generate_prompt(request: QueryRequest):
    """
    Enhances user queries using OpenAI's GPT model.
    """
    messages = [
        {"role": "system", "content": "You are an AI assistant helping users optimize their queries for language models. Only return the enhanced query without any additional explanations or introductory phrases."},
        {"role": "user", "content": f"Please enhance the following query to make it clearer and more effective for a language model: {request.message}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=messages,
            temperature=0.7,
            max_tokens=200,
            top_p=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.3
        )
        improved_query = response.choices[0].message.content.strip()
        
        return {"response": improved_query}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the service using: uvicorn prompt_engineering_service:app --host 0.0.0.0 --port 5007
