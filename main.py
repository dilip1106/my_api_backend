from fastapi import FastAPI, Request, Query
from fastapi.responses import FileResponse, PlainTextResponse
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Initialize Groq client
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def save_response_to_file(text: str, filename: str = "response.txt"):
    """Save model output to a text file and return its path."""
    file_path = os.path.join(os.getcwd(), filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text.strip())
    return file_path


@app.get("/generate")
async def generate_get(prompt: str = Query(...)):
    """Handle GET requests (via browser)."""
    try:
        response = client.responses.create(
            input=prompt,
            model="qwen/qwen3-32b"  # Best model for code/text generation
        )
        text = response.output_text.strip()

        file_path = save_response_to_file(text, "response.txt")
        return FileResponse(file_path, media_type="text/plain", filename="response.txt")

    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)


@app.post("/generate")
async def generate_post(request: Request):
    """Handle POST requests (via cURL or API)."""
    try:
        data = await request.json()
        prompt = data.get("prompt", "")

        if not prompt:
            return PlainTextResponse("Missing 'prompt' in request body", status_code=400)

        response = client.responses.create(
            input=prompt,
            model="qwen/qwen3-32b"
        )
        text = response.output_text.strip()

        file_path = save_response_to_file(text, "response.txt")
        return FileResponse(file_path, media_type="text/plain", filename="response.txt")

    except Exception as e:
        return PlainTextResponse(str(e), status_code=500)
