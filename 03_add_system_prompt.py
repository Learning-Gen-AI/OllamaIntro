import ollama
import pandas as pd

# Read the CSV file
data = pd.read_csv('file1.csv')

# Define system prompt for data analysis role
system_prompt = """You are a helpful expert CSV file analyzer with 20 years of experience. 
Answer the questions phrased to you professionally and to-the-point.
All communication should be technical with a consistent neautral tone.
Do not hallucinate. If you don't know an answer then tell me you are unsure, do not invent facts to support your argument."""

# Create user prompt with dataset info
user_prompt = "Describe this dataset to me:\n\n"
user_prompt += "Field names and data types:\n"
for column in data.dtypes.items():
    user_prompt += f"- {column[0]}: {column[1]}\n"

# Send both prompts to the model
response = ollama.chat(model='llama3.1', messages=[
    {'role': 'system', 'content': system_prompt},
    {'role': 'user', 'content': user_prompt}
])
print(response['message']['content'])