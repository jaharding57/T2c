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
    system_prompt = """system_prompt = You are a multimodal Parsons problem generator.

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
      "_thoughts": [...],  // optional design thoughts for debugging
      "problem": "...",
      "solution_blocks": [ { "id": "...", "code": "..." }, ... ],
      "distractor_blocks": [ { "id": "...", "code": "..." }, ... ],
      "difficulty": "Medium",  // optional
      "concepts": ["..."],     // optional
      "mode": "JavaScript"     // repeated per-problem
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

    # For C# batch
    task_spec1 = {
        "mode": "C#",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False}
        },
        "num_problems": 3
    }

    # For JavaScript batch
    task_spec2 = {
        "mode": "JavaScript",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": True},
            "Medium": {"Functions": True},
            "Hard": {"Recursion": False}
        },
        "num_problems": 1
    }

    # For Python + recursion batch
    task_spec3 = {
        "mode": "Python",
        "concepts": {
            "Easy": {"Variable Assignment": True, "Basic Arithmetic": False},
            "Medium": {"Functions": False},
            "Hard": {"Recursion": True}
        },
        "num_problems": 1
    }

    # For Smurfs batch
    task_spec_smurfs = {
        "mode": "Smurfs",
        "num_problems": 2
    }

    # For Gibberish batch
    task_spec_gibberish = {
        "mode": "Gibberish",
        "num_problems": 2
    }

    # Choose which batch to generate
    # deltas = generate_problems(task_spec_smurfs)
    deltas = generate_problems(task_spec_gibberish)


    # Collect and pretty-print the output
    import sys

    output_chunks = []
    for delta in deltas:
        if delta:
            output_chunks.append(delta)

    full_json_str = "".join(output_chunks)

    # Parse it back into a dict
    try:
        parsed = json.loads(full_json_str)
    except json.JSONDecodeError as e:
        print("❌ JSON parsing failed:", e)
        sys.exit(1)

    # Pretty-print to terminal
    print(json.dumps(parsed, indent=2))

    # (Optional) Save to file    

    # output file name(s)
    # op_file_nm = "smurfs_batch1.json" 
    op_file_nm = "gibberish_batch1.json"

    with open(op_file_nm, "w") as f:
        json.dump(parsed, f, indent=2)
        print("✅ Saved to smurfs_batch1.json")

