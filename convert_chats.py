import pandas as pd
import json

# Load the JSON file
file_path = '/Users/winter/Downloads/chats-2024-10-03.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Extract relevant data for conversion
chats_data = data['chats']

# Extract relevant data for conversion including 'model' and 'title'
flattened_data = []
for chat in chats_data:
    chat_model = chat.get('model', '')
    chat_title = chat.get('title', '')
    for message in chat.get('messages', []):
        flattened_data.append({
            'Chat ID': chat.get('_id', ''),
            'Message ID': message.get('id', ''),
            'Role': message.get('role', ''),
            'Content': message.get('content', ''),
            'Model': chat_model,
            'Title': chat_title,
            'Timestamp': message.get('timestamp', '')
        })

# Convert to DataFrame
df = pd.DataFrame(flattened_data)

# Save as CSV
csv_file_path_with_model = '/Users/winter/Downloads/chats_with_model_and_title.csv'
df.to_csv(csv_file_path_with_model, index=False)

print(csv_file_path_with_model)
