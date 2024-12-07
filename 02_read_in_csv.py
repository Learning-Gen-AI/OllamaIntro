import ollama
import pandas as pd

# Read the CSV file
data = pd.read_csv('file1.csv')

# Create a description of the dataset
prompt = "Describe this dataset to me:\n\n"
prompt += "Field names and data types:\n"
for column in data.dtypes.items():
    prompt += f"- {column[0]}: {column[1]}\n"

response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': prompt}])
print(response['message']['content'])