#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = ["openai", "python-dotenv"]
# ///

import json
import os

from openai import OpenAI
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

def generate_problems(task_specification):
    """
    Generate programming problems based on a task specification.

    Args:
        task_specification (dict): A dictionary containing the task specification.

    Returns:
        dict: A JSON object containing the generated problems.
    """
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
    chunks = client.chat.completions.create(
        model=openai_api_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(task_specification)},
        ],
        response_format={"type": "json_object"},
        stream=True,
    )

    for chunk in chunks:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


if __name__ == "__main__":

    # for batch 1
    task_spec1 = {
        "language": "C+",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False},
        },
        "num_problems": 3,
    }

    # for batch 2
    task_spec2 = {
        "language": "Javascript",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False},
        },
        "num_problems": 1,
    }

    # for batch 3
    task_spec3 = {
        "language": "Python",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": False},
            "Medium": {"Functions": False},
            "Hard": {"Recursion": True},
        },
        "num_problems": 1,
    }

    # for batch 4
    task_spec = {
        "language": "Python",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False},
        },
        "num_problems": 3,
    }

    # Generate problems
    deltas = generate_problems(task_spec3)

    # Print the streaming output
    for delta in deltas:
        print(delta, end="")
