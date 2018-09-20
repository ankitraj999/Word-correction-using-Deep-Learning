import os

modifying_chars = 'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM .,'

max_len_text = 30
wrong_sent_len = 33

project_path = os.path.dirname(os.path.realpath(__file__))

unprocessed_data = os.path.join(project_path, 'data/unprocessed')
processed_data = os.path.join(project_path, 'data/processed')

allowed_chars = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM .,"+"'"
char_to_index = {}
index_to_char = {}
for i, char in enumerate(allowed_chars):
    char_to_index[char] = i+1
    index_to_char[i+1] = char

allowed_chars = set(allowed_chars)


unprocessed_file_names = [os.path.join(unprocessed_data, f) for f in os.listdir(unprocessed_data)]
processed_file_names = [os.path.join(processed_data, f) for f in os.listdir(processed_data)]