import os
from dotenv import load_dotenv
import pandas as pd
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Load Hugging Face token
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

# Log in using the Hugging Face token
login(token=huggingface_token)

# Load model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name, use_auth_token=huggingface_token)
model = GPT2LMHeadModel.from_pretrained(model_name, use_auth_token=huggingface_token)
model.eval()

device = torch.device("cpu")
model.to(device)

# Function to generate counterspeech
def generate_counterspeech(hate_speech: str) -> str:
    prompt = f"You are tasked with analyzing this hate speech and responding with a logical, respectful, and concise counterspeech of up to 4 sentences using factual and statistical evidence when possible. Hate speech: {hate_speech}"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=75,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if prompt in generated_text:
        counterspeech = generated_text[len(prompt):].strip()
    else:
        counterspeech = generated_text.strip()

    if "\n\n" in counterspeech:
        counterspeech = counterspeech.split("\n\n")[0].strip()

    return counterspeech

# Load hate speech CSV
csv_path = "hatespeech_data.csv"
df = pd.read_csv(csv_path)
unique_hate_speech = df["HATE_SPEECH"].dropna().unique()

output_csv = "data.csv"

# Try to load previously saved counterspeech
try:
    existing_df = pd.read_csv(output_csv)
    existing_hate_speech = existing_df["hate_speech"].unique()
except FileNotFoundError:
    existing_hate_speech = []

# The number of hate speeches the pprogram should process
max_to_process = 3
count = 0

# Open CSV for writing
with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Write header if file is new
    if not os.path.exists(output_csv) or os.stat(output_csv).st_size == 0:
        writer.writerow(["hate_speech", "generated_counterspeech"])

    for hate_speech in unique_hate_speech:
        if count >= max_to_process:
            break

        if hate_speech in existing_hate_speech:
            print(f"Skipping duplicate: {hate_speech}")
            continue

        print(f"Processing hate speech: {hate_speech}")
        counterspeech = generate_counterspeech(hate_speech)
        print(f"Counterspeech: {counterspeech}\n")
        writer.writerow([hate_speech, counterspeech])
        file.flush()
        count += 1

print("Done processing.")
