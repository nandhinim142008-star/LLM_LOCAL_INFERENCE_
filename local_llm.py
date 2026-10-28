
import pandas as pd
import ollama
from pathlib import Path

# Dataset folder path
DATASET = Path("dataset")

# Load prompts
prompts = pd.read_csv(DATASET / "prompts.csv")

# Use the smaller model
model = "gemma3:1b"

responses = []

# Clear old conversation history
with open(DATASET / "conversation_history.txt", "w", encoding="utf-8") as file:
    file.write("Week 8 - Local LLM Inference\n\n")

# Generate responses
for index, row in prompts.iterrows():
    prompt = row["PROMPT"]

    print(f"Processing {index + 1}/{len(prompts)}: {row['DOMAIN']}")

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = response["message"]["content"]

    except Exception as error:
        answer = f"ERROR: {error}"
        print(answer)

    # Save conversation history
    with open(DATASET / "conversation_history.txt", "a", encoding="utf-8") as file:
        file.write(f"PROMPT_ID: {row['PROMPT_ID']}\n")
        file.write(f"DOMAIN: {row['DOMAIN']}\n")
        file.write(f"PROMPT: {prompt}\n")
        file.write(f"RESPONSE: {answer}\n")
        file.write("-" * 80 + "\n")

    responses.append({
        "PROMPT_ID": row["PROMPT_ID"],
        "DOMAIN": row["DOMAIN"],
        "PROMPT": prompt,
        "RESPONSE": answer
    })

# Save response files
result = pd.DataFrame(responses)

result.to_csv(DATASET / "llm_responses.csv", index=False)

comparison = result[["PROMPT_ID", "DOMAIN", "PROMPT", "RESPONSE"]]
comparison.to_csv(DATASET / "response_comparison.csv", index=False)

print("\nResponses saved successfully.")
print("Response comparison saved successfully.")
print("Conversation history saved successfully.")
print("Week 8 project completed!")
