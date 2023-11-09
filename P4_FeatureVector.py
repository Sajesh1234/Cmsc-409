import re
from Porter_Stemmer_Python import PorterStemmer
import csv

def tokenize(paragraph):
    return re.findall(r'\b\w+\b', paragraph)

def remove_special_characters(text):
    text = re.sub(r'(<br\s*/><br\s*/?>|<br><br/>)', ' ', text)  
    text = re.sub(r'[^a-zA-Z\s]', '', text)  
    return text

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def convert_to_lowercase(text):
    return text.lower()

def remove_stop_words(text, stop_words):
    return ' '.join([word for word in text.split() if word not in stop_words])


with open('Project4_stop_words.txt', 'r') as f:
    stop_words = set(f.read().splitlines())


with open('Project4_paragraphs.txt', 'r') as f:
    paragraphs = f.readlines()


porter = PorterStemmer()


feature_vectors = []


all_words = []

for paragraph in paragraphs:
    
    paragraph_feature = {}
    
    tokens = tokenize(paragraph)
    w = remove_special_characters(paragraph)
    w = remove_numbers(w)
    w = convert_to_lowercase(w)
    w = remove_stop_words(w, stop_words)
    stemmed_words = [porter.stem(word, 0, len(word)-1) for word in w.split()]
    
    
    all_words.extend(stemmed_words)
    
    
    for word in stemmed_words:
        if word in paragraph_feature:
            paragraph_feature[word] += 1
        else:
            paragraph_feature[word] = 1
            
    feature_vectors.append(paragraph_feature)

# Print the feature vector in a tabular format
with open('feature_vector_output.txt', 'w') as file:
    file.write("Feature Vector [")
    for i, word in enumerate(set(all_words)):
        if i == 0:
            file.write(f"'{word}'")
        else:
            file.write(f",'{word}'")
    file.write("]\n")
