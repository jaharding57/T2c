from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.requests import Request
from fastapi.exceptions import HTTPException
import os
import json
import openai
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
import base64

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API base and model name from environment variables
openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
openai_api_model_name = os.getenv("OPENAI_API_MODEL_NAME", "gpt-4o")

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please add your OpenAI API key to the .env file.")

# Configure OpenAI API
client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/generate-problems")
def generate_problems(specification: str):
    try:
        decoded_spec = base64.b64decode(specification).decode("utf-8")

        system_prompt = """You are a multimodal Parsons problem generator.

Your task is to generate a set of block-based ordering problems in a selected mode. A "mode" represents a topical or syntactic domain (e.g., Python code, gibberish lines, Smurf facts, etc.). Each problem should consist of:
- a problem description,
- a sequence of correct blocks in order (solution_blocks),
- and a set of distractor blocks that seem plausible but are incorrect, unordered, or irrelevant.

The output should be a JSON object structured as follows:

{
  "mode": "JavaScript", // or "Python", "C#", "Gibberish", or "Smurfs"
  "num_problems": 3,
  "problems": [
    {
      "_thoughts": [...],
      "problem": "...",
      "solution_blocks": [ { "id": "...", "code": "..." }, ... ],
      "distractor_blocks": [ { "id": "...", "code": "..." }, ... ],
      "difficulty": "Medium",
      "concepts": ["..."],
      "mode": "JavaScript"
    },
    ...
  ]
}

Supported modes:

1. **Python**, **JavaScript**, and **C#**:
   - Generate concise, valid code snippets (functions, conditionals, loops, variable assignments).
   - Distractors should be syntactically valid but semantically incorrect, misleading, or incomplete.
   - Maintain typical formatting and idioms for each language.

2. **Gibberish**:
   - Generate lines that resemble nonsense or whimsical code (e.g., `glorp = bleep(9);`)
   - The correct order should follow a fake logical structure.
   - Distractors can look syntactically valid but disrupt or confuse the intended nonsense sequence.

3. **Smurfs**:
   - Generate factual-sounding statements about Smurfs.
   - `solution_blocks` should represent **true** Smurf-related events or facts in **chronological order**.
   - Distractors should include:
     - real facts in the wrong order,
     - incorrect or distorted facts,
     - or unrelated trivia.

**Important:** Do not include identical lines in both solution_blocks and distractor_blocks.

**Your output must contain exactly as many problems as specified in the input task specification.**

Return only the JSON structure. Do not include any commentary or explanation outside of the JSON object.
"""

        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": decoded_spec}
            ],
            response_format={"type": "json_object"}
        )

        ai_response = response.choices[0].message.content
        problems = json.loads(ai_response)

        return JSONResponse(content=problems)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating problems: {str(e)}")
