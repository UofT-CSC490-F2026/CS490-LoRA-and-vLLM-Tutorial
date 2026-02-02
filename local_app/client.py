from openai import OpenAI
import sys

# Connect to local vLLM server
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def query_model(model_id, prompt, max_tokens=100, temperature=0.7):
    print(f"\n--- Querying {model_id} ---")
    print(f"Prompt: {prompt[:50]}...")
    
    try:
        response = client.completions.create(
            model=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["###"] # Stop generation at next instruction header
        )
        print("Response:")
        print(response.choices[0].text.strip())
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure the server is running on localhost:8000")

if __name__ == "__main__":
    # 1. Test SQL Adapter
    sql_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Get all users from Canada.

### Input:


### Response:
"""
    query_model("sql-model", sql_prompt, temperature=0.1)

    # 2. Test Shakespeare Adapter
    shake_prompt = "### Style: Shakespeare\n### Content: Explain the concept of love."
    query_model("shakespeare-model", shake_prompt, temperature=0.7)