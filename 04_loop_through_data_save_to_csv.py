import ollama
import pandas as pd

# Read the CSV file
data = pd.read_csv('file1.csv')

# Get column names and types
headers_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])

# Define system prompt
system_prompt = """You are a helpful expert CSV file analyzer with 20 years of experience. 
Answer the questions phrased to you professionally and to-the-point.
All communication should be technical with a consistent neautral tone.
Do not hallucinate. If you don't know an answer then tell me you are unsure, do not invent facts to support your argument."""

# Create list to store LLM descriptions
llm_descriptions = []

# Process each row
for index, row in data.iterrows():
    # Convert row to string format
    row_string = ', '.join([f"{col}: {val}" for col, val in row.items()])
    
    # Create contextual prompt
    contextual_prompt = f"""Dataset columns and their types:
{headers_info}

Record to analyze:
{row_string}"""
    
    # Get LLM response for this record
    response = ollama.chat(model='llama3.1', messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': f"Describe this record and flag any values that look like they might be incorrect and need human review: {contextual_prompt}"}
    ])
    
    # Append the LLM's description to our list
    llm_descriptions.append(response['message']['content'])

# Add descriptions as new column
data['llm_description'] = llm_descriptions

# Save enriched dataset
data.to_csv('enriched_data.csv', index=False)