import os
import random
import data_config
import editdistance


'''
code to introducing spelling errors to a given sentence
four types of spelling mistakes
toss = 0 --> replace one character
toss = 1 --> remove one character
toss = 2 --> insert one character
toss = 3 --> interchance positions of two next to next characters
'''


def wrong_text_gen(text, min_edit_ratio=0.08):
    original_text = text
    while float(editdistance.eval(original_text, text))/len(original_text) < min_edit_ratio:
        toss = random.randint(0,3)
        try:
            if toss == 0:
                char_posi = random.randint(0, len(text)-1)
                text = text[:char_posi] + random.choice(data_config.modifying_chars) + text[char_posi+1:]
            if toss == 1:
                char_posi = random.randint(0, len(text)-1)
                text = text[:char_posi] + text[char_posi+1:]
            if toss == 2:
                char_posi = random.randint(0, len(text))
                text = text[:char_posi] + random.choice(data_config.modifying_chars) + text[char_posi:]
            if toss == 3:
                char_posi = random.randint(0, len(text)-2)
                text = text[:char_posi] + text[char_posi+1] + text[char_posi] + text[char_posi+2:]
        except Exception as e:
            print(e)
    return text

def preprocess(text):
    text = text.strip()
    
    text =text.split()
    list_text=[]
    for word in text:
        processed_text = ''
        for char in word:
            if char in data_config.allowed_chars:
                processed_text = processed_text + char
        list_text.append(processed_text)
    return list_text


'''
Function to read each file and write a new file in the following format
wrong_text_1 + '\t' + correct_text_2
wrong_text_2 + '\t' + correct_text_2
here, wrong text is the input to our model and correct text is the output
'''
def data_generator(file_names):
    for file_name in file_names:
        processed_data_path = os.path.join(data_config.processed_data, file_name.split('/')[-1])
        f = open(processed_data_path, 'w')
        print("Processed data path:", processed_data_path)
        lines = open(file_name).readlines()
        print("Total:", file_name, len(lines))
        for i, line in enumerate(lines):
            if i%1000 == 0:
                print("Current:", file_name, i)
            line1 = preprocess(line)
            
            for words in line1:
                if len(words) <= data_config.max_len_text and len(words) > 1:
                    wrong_words = wrong_text_gen(words)
                    f.write(wrong_words + '\t' + words + '\n')
        f.close()


    