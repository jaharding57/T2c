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
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
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
        # Decode the base64-encoded specification
        decoded_spec = base64.b64decode(specification).decode("utf-8")

        # Construct the system prompt
        system_prompt = """You are a Parsons problem generator.

Your task is to generate a set of programming problems in the selected language, relevant to the selected concepts and difficulties. Each problem should include:
- a clear problem description,
- a set of solution blocks (in correct order),
- a set of distractor blocks (plausible but incorrect or out-of-place code),
- and optional metadata such as difficulty and concepts.

The output must be a JSON object with this structure:

{
  "language": "Python",
  "num_problems": 3,
  "problems": [
    {
      "_thoughts": [
        "This problem focuses on using a for-loop to iterate and update a variable.",
        "I will include a distractor with the wrong range and another with an unrelated variable."
      ],
      "problem": "Arrange the code so that x == 3 when the program terminates.",
      "solution_blocks": [
        { "id": "block-a1", "code": "x = 27" },
        { "id": "block-b1", "code": "for i in range(2):" },
        { "id": "block-c1", "code": "x /= 3" }
      ],
      "distractor_blocks": [
        { "id": "block-d1", "code": "y = 57" },
        { "id": "block-e1", "code": "x *= y" },
        { "id": "block-f1", "code": "for i in range(3):" }
      ],
      "difficulty": "Medium",
      "concepts": ["Loops", "Variable Assignment"],
      "language": "Python"
    },
    ...
  ]
}

**Each `solution_blocks` and `distractor_blocks` list must contain objects, each with:**
- `"id"`: a unique identifier string like `"block-a1"`, `"block-b1"`, etc.
- `"code"`: a single line of code.

The problems should be relevant only to the selected concepts and reflect the indicated difficulty levels. Do not include concepts that were marked false.

The `"num_problems"` field in the output must match the `num_problems` value from the input task specification.
IMPORTANT: DO NOT INCLUDE DUPLICATE CODE SNIPPETS ACROSS SOLUTIONS AND DISTRACTORS...

Return a well-formed JSON object exactly as described above. Do not include any explanation, extra commentary, or surrounding prose — just the JSON.
"""

        # Call OpenAI API
        response = client.chat.completions.create(
            model=openai_api_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": decoded_spec}
            ],
            response_format={"type": "json_object"}
        )

        # Extract the AI-generated content
        ai_response = response.choices[0].message.content

        # Parse the AI response into JSON
        problems = json.loads(ai_response)

        return JSONResponse(content=problems)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating problems: {str(e)}")
