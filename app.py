# Import necessary libraries for building a FastAPI web application
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import requests
import os
import json
import asyncio

# Create the main FastAPI application instance
app = FastAPI()

# Mount static files directory to serve frontend assets
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration settings for Ollama AI model API
ollama_url = "http://localhost:11434/api/generate"
ollama_model = "tinyllama"

@app.get("/")
def serve_homepage():
    return FileResponse(os.path.join("static", "index.html"))

@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon configured"}

@app.post("/chat")
async def chat(prompt: str = Query(..., description="User prompt for AI model")):
    async def generate_stream():
        try:
            # 🧠 Updated: Strong system prompt to guide clean, accurate, relevant responses
            custom_prompt = (
                "You are a smart, factual, and precise AI assistant. "
                "Always answer based only on the user's request. "
                "Do NOT explain what you can do. Do NOT repeat the question. "
                "If the user asks for a list or explanation, respond with clean, numbered points like:\n"
                "1. First point\n2. Second point\n3. ...\n\n"
                "Use simple language. Avoid unnecessary introductions like 'Sure!' or 'I'm happy to help.'\n\n"
                f"User: {prompt}\nAI:"
            )

            response = requests.post(
                ollama_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": ollama_model,
                    "prompt": custom_prompt,
                    "stream": True,
                    "options": {
                        "num_predict": 180,     # More tokens for longer answers
                        "temperature": 0.2      # More focused and factual
                    }
                },
                stream=True,
                timeout=30
            )

            if not response.ok:
                yield f"data: {json.dumps({'error': f'HTTP error! status: {response.status_code}'})}\n\n"
                return

            for line in response.iter_lines():
                if line:
                    try:
                        line_str = line.decode('utf-8')
                        json_response = json.loads(line_str)

                        if 'response' in json_response:
                            response_text = json_response['response']
                            yield f"data: {json.dumps({'word': response_text})}\n\n"
                            await asyncio.sleep(0.05)

                        if json_response.get('done', False):
                            yield f"data: {json.dumps({'done': True})}\n\n"
                            break
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            yield f"data: {json.dumps({'error': f'Error connecting to Ollama API: {str(e)}'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'An unexpected error occurred: {str(e)}'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
