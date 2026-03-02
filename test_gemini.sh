#!/bin/bash

# Configuration
PROJECT_ID="bypass-481809"
LOCATION="global"
MODEL_ID="gemini-3.1-pro-preview"
API_ENDPOINT="https://aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${LOCATION}/publishers/google/models/${MODEL_ID}:streamGenerateContent"

echo "Testing connection to ${MODEL_ID}..."

# Loop for retries (Exponential Backoff)
for i in {1..4}; do
  RESPONSE=$(curl -s -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "$API_ENDPOINT" \
    -d '{
      "contents": [{
        "role": "user",
        "parts": [{"text": "Briefly describe the current state of the S&P 500."}]
      }],
      "generationConfig": {
        "temperature": 0.7,
        "maxOutputTokens": 200
      }
    }')

  # Check if the response contains an error
  if [[ $RESPONSE == *"error"* ]]; then
    CODE=$(echo $RESPONSE | grep -o '"code": [0-9]*' | cut -d' ' -f2)
    echo "Attempt $i failed (Status $CODE). Retrying in $((2**i)) seconds..."
    sleep $((2**i))
  else
    echo "--- Success ---"
    # Use python to pretty-print the JSON if available
    echo $RESPONSE | python3 -m json.tool
    exit 0
  fi
done

echo "Test failed after 4 attempts."
