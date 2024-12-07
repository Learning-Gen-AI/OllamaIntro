import ollama

response = ollama.chat(model='llama3.1', messages=[{'role': 'user', 'content': 'What colors are contained in the American flag?'}])

print(response['message']['content'])