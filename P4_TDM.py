import re
from Porter_Stemmer_Python import PorterStemmer
import csv
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from minisom import MiniSom

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
    paragraphs = f.read().split('\n\n') 

porter = PorterStemmer()

all_words = set()

word_counts = []


for paragraph in paragraphs:
    tokens = tokenize(paragraph)
    w = remove_special_characters(paragraph)
    w = remove_numbers(w)
    w = convert_to_lowercase(w)
    w = remove_stop_words(w, stop_words)
    stemmed_words = [porter.stem(word, 0, len(word)-1) for word in w.split()]
    word_count = {}
    for word in stemmed_words:
        if word not in word_count:
            word_count[word] = 0
        word_count[word] += 1
        all_words.add(word)
        
    word_counts.append(word_count)

with open('TDM.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    header = ['Paragraph'] + list(all_words)
    csv_writer.writerow(header)

    for i, word_count in enumerate(word_counts, 1):
        row = [f'Paragraph {i}']
        for word in all_words:
            if word in word_count:
                row.append(word_count[word])
            else:
                row.append(0)
        csv_writer.writerow(row)

data = pd.read_csv('TDM.csv')


scaler = MinMaxScaler()
datax = data.drop('Paragraph', axis=1)
datax_normalized = scaler.fit_transform(datax)


som = MiniSom(3, 3, datax_normalized.shape[1], sigma=0.3, learning_rate=0.5)
som.train_random(datax_normalized, 100) 



somwinners = []
for i in range(datax_normalized.shape[0]):
    somwinners.append(som.winner(datax_normalized[i]))


data['Cluster'] = somwinners



unique_winners = set(somwinners)
for idx, winner in enumerate(unique_winners):
    cluster_paragraphs = data[data['Cluster'] == winner].index
    print(f"Cluster {idx + 1} Paragraphs:")
    for paragraph in cluster_paragraphs:
        print("paragraph", paragraph)
    print("\n______________\n")
