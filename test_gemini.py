import os
# Ensure gcloud is happy even inside Python's subprocess calls
os.environ["CLOUDSDK_PYTHON"] = "/opt/homebrew/bin/python3.11"

from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project='bypass-481809', location='global')

# Higher token limit for detailed stock reports
config = types.GenerateContentConfig(
    max_output_tokens=2048,
    temperature=0.2, # Lower temperature is better for financial data
)

response = client.models.generate_content(
    model='gemini-3.1-pro-preview',
    contents='Give me a technical analysis of PYPL and LCID.',
    config=config
)

print(f"Thought process length: {response.usage_metadata.thoughts_token_count}")
print(f"Final Answer: {response.text}")
