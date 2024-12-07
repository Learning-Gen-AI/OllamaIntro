import ollama
import pandas as pd

# Read the CSV file
data = pd.read_csv('file1.csv')

# Get column names and types
headers_info = "\n".join([f"{col}: {dtype}" for col, dtype in data.dtypes.items()])

# Define system prompt with structured output requirement
system_prompt = """You are a helpful expert CSV file analyzer with 20 years of experience. For each record, provide two pieces of information in exactly this format:

DESCRIPTION: [your single-sentence description of the record]
FLAGS: [comma-separated list of any concerning values that need human review, or NONE if nothing is flagged]

Answer the questions phrased to you professionally and to-the-point.
All communication should be technical with a consistent neautral tone.
Be concise and factual. Always maintain this exact format with these exact labels.
Do not hallucinate. If you don't know an answer then tell me you are unsure, do not invent facts to support your argument.
"""

# Create lists to store LLM outputs
record_descriptions = []
flagged_values = []

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
    
    # Parse the response
    llm_response = response['message']['content']
    description = ""
    flags = "NONE"
    
    # Extract description and flags from response
    for line in llm_response.split('\n'):
        if line.startswith('DESCRIPTION:'):
            description = line.replace('DESCRIPTION:', '').strip()
        elif line.startswith('FLAGS:'):
            flags = line.replace('FLAGS:', '').strip()
    
    # Append to our lists
    record_descriptions.append(description)
    flagged_values.append(flags)

# Add new columns to the dataframe
data['RecordDescription'] = record_descriptions
data['FlaggedValues'] = flagged_values

# Save enriched dataset
data.to_csv('enriched_data.csv', index=False)