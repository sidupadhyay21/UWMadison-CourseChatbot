import tiktoken
import json

# Load the JSON file
# Need to be navigated into the correct directory
with open('UWCourseCatalog_05-23-2024.json', 'r') as file:
    data = json.load(file)
    json_string = json.dumps(data)
    file.close()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string(json_string, "cl100k_base"))