#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 16:03:23 2020

@author: borisnorgaard
"""

import warnings
from collections import defaultdict

import en_core_web_md
import gensim
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nltk
import numpy as np
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from sklearn.manifold import TSNE
from tqdm import tqdm

warnings.filterwarnings("ignore")
spacy_model = en_core_web_md.load()

# Importation du texte

HP = 'data/Harry-Potter-and-the-Sorcerer.txt'

# Détections des phrases via NLTK

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')

all_sentences = []
with open(HP, "r", encoding='unicode_escape') as fp:
    text = fp.read()
    text = text.replace("\n\n", " ")
    all_sentences = sent_detector.tokenize(text.strip())  # nltk feels to be better in detecing sentences, bu
print(len(all_sentences))


# Création fonction qui simplifie la ponctuation
def delete_punctuation(sent):
    sent = sent.replace("“", "")  # Weird character
    sent = sent.replace("’s", "")  # Weird character 2
    sent = sent.replace("’", "")  # Weird character 3
    sent = sent.replace("”", "")  # Weird character 4
    sent = sent.replace("‘", "")  # Weird character 5
    #     words = sent.split()
    #     table = str.maketrans('', '', string.punctuation)
    #     stripped = [w.translate(table) for w in words]
    #     sent_clean = " ".join(stripped)
    return sent


# Traitement du texte pour créer une liste de personnages
person_set = set([])
person_dict = defaultdict(int)
location_set = set([])

all_sentences_clean = []
for sent in tqdm(all_sentences):
    sent_clean = delete_punctuation(sent)
    doc = spacy_model(sent_clean)
    for X in doc.ents:
        if X.label_ == "PERSON":
            person_dict[X.text] += 1
            person_set.add(X.text)
        if X.label_ == "LOC":
            location_set.add(X.text)
    all_sentences_clean.append(sent_clean)
print(f"FOUND: Persons :: {len(person_set)}, Locations :: {len(location_set)}")

person_dict = {k: v for k, v in person_dict.items() if v != 1}

# Suppression des non-personnages    

del person_dict["yeh"]
del person_dict["Quidditch"]
del person_dict["Quaffle"]
del person_dict["Stone"]
del person_dict["Pince"]
del person_dict["Slytherin"]
del person_dict["Muggle"]
del person_dict["Slytherins"]
del person_dict["Hufflepuff"]
del person_dict["Gryffindor"]
del person_dict["Seeker"]
del person_dict["Ravenclaw"]
del person_dict["Gringotts"]
del person_dict["Scabbers"]
del person_dict["Remembrall"]
del person_dict["Snitch"]
del person_dict["Smeltings"]
del person_dict["knowin"]
del person_dict["don"]
del person_dict["Don"]
del person_dict["Galleons"]
del person_dict["Oy"]
del person_dict["Sprout"]
del person_dict["Harry sat"]
del person_dict["Dursleys"]
del person_dict["Yeh"]
del person_dict["Hogwarts"]
del person_dict["yeh  "]
del person_dict["Nah"]
del person_dict["D'yeh"]
del person_dict["tremblin"]
del person_dict["Harry Potter's"]
del person_dict["Yer"]
del person_dict["Harry shivered"]
del person_dict["Diagon Alley"]
del person_dict["Bye"]
del person_dict["Agrippa"]
del person_dict["Ickle Firsties"]
del person_dict["Malfay"]
del person_dict["McGonagall Harry"]
del person_dict["Bludger"]
del person_dict["Charms"]
del person_dict["Wingardium Leviosa"]
del person_dict["bush"]
del person_dict["Harry nodded"]

# Création liste de personnages à la main sur base des résultats trouvés précédemment

manual_character_list = {"Dursleys": ["Dursleys"],
                         "Harry": ["Harry Potter", "H. Potter", "Potter", "Harry Potter", "Mr. Potter"],
                         "Dumbledore": ["Albus Dumbledore"], "Vernon": ["Uncle Vernon", "Dursley"],
                         "Petunia": ["Aunt Petunia"],
                         "Ron": ["Weasley", "Ronald Weasley"], "Hermione": ["Hermione Granger", "Miss Granger"],
                         "Malfoy": ["Draco Malfoy", "Draco"], "Hagrid": ["Hagrid"], "Snape": ["Severus"],
                         "Quirrell": ["Quirrell"], "Dudley": ["Dudley"], "Neville": ["Neville", "Neville Longbottom"],
                         "McGonagall": ["McGonagall"], "Voldemort": ["Voldemort"], "Filch": ["Filch"],
                         "Percy": ["Percy"], "George": ["George"], "Fred": ["Fred"], "Flamel": ["Flamel"],
                         "Fluffy": ["Fluffy"], "Goyle": ["Goyle"], "Norbert": ["Norbert"], "Charlie": ["Charlie"],
                         "Crabbe": ["Crabbe"], "Flitwick": ["Flitwick"], "Fang": ["Fang"], "Firenze": ["Firenze"],
                         "Ollivander": ["Ollivander"], "Bane": ["Bane"], "Seamus": ["Seamus"], "Pomfrey": ["Pomfrey"],
                         "Norris": ["Norris"], "Hedwig": ["Hedwig"], "Hooch": ["Hooch"], "Lee": ["Lee"],
                         "Piers": ["Piers"], "Dean": ["Dean"], "Flint": ["Flint"], "Figg": ["Figg"],
                         "Malkin": ["Malkin"], "Scabbers": ["Scabbers"]}

persons_list = sorted(list(person_dict.items()), key=lambda x: x[1], reverse=True)

# Importation du réseau du projet 1


# Traitement du texte pour préparer à Word2Vec

all_sentences_clean2 = []
for sent in tqdm(all_sentences_clean):
    sent_clean = sent
    for x, y in manual_character_list.items():
        for k in y:
            sent_clean = sent_clean.replace(k, x)
    all_sentences_clean2.append(sent_clean)

CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, remove_stopwords, strip_multiple_whitespaces]

all_sentences_preprocessed = []
for sent in tqdm(all_sentences_clean2):
    parsed_line = preprocess_string(sent, CUSTOM_FILTERS)
    all_sentences_preprocessed.append(parsed_line)

# DEFINE MODEL
model = gensim.models.Word2Vec(size=100, window=4,
                               min_count=10, alpha=0.01)

# BUILD VOCABULARY
model.build_vocab(all_sentences_preprocessed)

# AND TRAIN THE MODEL
iterations = tqdm(range(10))
for i in iterations:
    model.train(all_sentences_preprocessed, total_examples=model.corpus_count, compute_loss=True,
                epochs=100)
    msg = f"Iter :: {i} -- Loss :: {model.get_latest_training_loss()}"
    iterations.set_postfix_str(s=msg, refresh=True)

# GET THE VOCABULARY FROM THE MODEL
vocabulary = list(model.wv.vocab)

# GET THE WORD EMBEDDING VECTORS
embedding_vectors = model[model.wv.vocab]

# GET THE LOWER CASE REPRESENTATION OF NAMED ENTITIES THAT ARE IN THE VOCABULARY
person_set_processed = [s for s in manual_character_list.keys() if s.lower() in vocabulary]
person_set_vocab = [s.lower() for s in person_set_processed]

# TRANSFORM THE EMBEDDING USING T-SNE INTO 2D
V_tranform = TSNE(n_components=2).fit_transform(embedding_vectors)

# PLOT THE PROJECTION
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*V_tranform), marker='.', s=50, lw=0, alpha=0.7,
            edgecolor='k')
# for i, (x,y) in enumerate(V_tranform):
#     plt.text(x,y, vocabulary[i], 
#                 fontsize = 4, alpha = 0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# THIS IS THE IMPLEMENTATION OF THE KMEANS WITH COSINE DISTANCE
from nltk.cluster import KMeansClusterer

# desired number of clusters to find
NUM_CLUSTERS = 4

# Sparial clustering with k-means

kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=10)

# be careful to supply the original vectors to the algorithm!
assigned_clusters = kclusterer.cluster(embedding_vectors, assign_clusters=True)

# DEFINE COLORS OF CLUSTERS 
colors = cm.nipy_spectral(np.array(assigned_clusters).astype(float) / NUM_CLUSTERS)

# PLOT THE RESULTS
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*V_tranform), marker='.', s=50, lw=0, alpha=0.7, c=colors,
            edgecolor='k')

for i, (x, y) in enumerate(V_tranform):
    plt.text(x, y, vocabulary[i], color=colors[i],
             fontsize=4, alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# THIS IS THE IMPLEMENTATION OF THE KMEANS WITH EUCLIDEAN DISTANCE
from sklearn.cluster import KMeans

# desired number of clusters to find
NUM_CLUSTERS = 4

# Sparial clustering with k-means

kclusterer_sklearn = KMeans(n_clusters=NUM_CLUSTERS)

# be careful to supply the projected vectors (2D) to the algorithm!
assigned_clusters = kclusterer_sklearn.fit_predict(V_tranform)

# DEFINE COLORS OF CLUSTERS 
colors = cm.nipy_spectral(np.array(assigned_clusters).astype(float) / NUM_CLUSTERS)

# PLOT THE RESULTS
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*V_tranform), marker='.', s=50, lw=0, alpha=0.7, c=colors,
            edgecolor='k')

for i, (x, y) in enumerate(V_tranform):
    plt.text(x, y, vocabulary[i], color=colors[i],
             fontsize=4, alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# GET SUBSET OF VECTORS
person_embedding_vectors = model[person_set_vocab]

# PROJECT IT INTO 2D
V_tranform = TSNE(n_components=2).fit_transform(person_embedding_vectors)

# PLOT THE PROJECTION
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*V_tranform), marker='.', s=100, lw=0, alpha=0.7,
            edgecolor='k')
for i, (x, y) in enumerate(V_tranform):
    plt.text(x, y, person_set_vocab[i],
             fontsize=12, alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.show()

# Word Similarity
word = "harry"
similar_words = sorted(model.wv.most_similar(positive=[word], topn=10), key=lambda x: x[1], reverse=False)
X, Y = zip(*similar_words)
X = [x.upper() for x in X]
colors = plt.cm.jet(np.linspace(0, 1, len(X)))
fig = plt.figure(figsize=(3, 6), dpi=120)
plt.title(f"Most similar to :: {word.upper()}")
plt.barh(X, Y, align='center', height=0.75, color=colors)
plt.xlabel("Cosine similarity")
plt.show()


# Avec le network

def present_in(noun, para):
    b = text_tokenize(para)
    for word in b:
        if noun == word:
            return True
    return False


def presence_in_hp(noun):
    presence = []
    counter = 0
    for para in paragraphe:
        if present_in(noun, para):
            presence.append(counter)
        counter += 1
    return presence
