"""
Created on Fri Nov 13 2020

@author: Boris Norgaard, Nils Boulanger & Martin Beaufayt
"""

import warnings
from collections import defaultdict

import en_core_web_md
import gensim
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import itertools

from sklearn.cluster import KMeans
from community import community_louvain
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import remove_stopwords
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_non_alphanum
from nltk.cluster import KMeansClusterer
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

# %% TEXT PROCESSING
HP = 'data/Harry-Potter-and-the-Sorcerer.txt'  # Harry Potter import

# Split text into sentences
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
with open(HP, "r", encoding='unicode_escape') as fp:
    text = fp.read()
    one_paragraphe = text.split("\n")  # ligne pour obtenir une variable paragraphe
    text = text.replace("\n\n", " ")
    all_sentences = sent_detector.tokenize(text.strip())  # nltk feels to be better in detecing sentences


def delete_punctuation(sent):
    """ Function wich simplifies the punctuation in a given sentence
    :param sent: input sentence
    :return: cleaned input sentence
    """
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


# Variable definition
person_set = set([])
person_dict = defaultdict(int)
location_set = set([])

warnings.filterwarnings("ignore")
spacy_model = en_core_web_md.load()

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
manual_character_dic = dict(Harry=["Harry", "Harry Potter", "H. Potter", "Potter", "Harry Potter", "Mr. Potter"],
                            Dumbledore=["Dumbledore", "Albus Dumbledore"], Vernon=["Vernon", "Uncle Vernon", "Dursley"],
                            Petunia=["Petunia", "Aunt Petunia"], Ron=["Ron", "Weasley", "Ronald Weasley"],
                            Hermione=["Hermione", "Hermione Granger", "Miss Granger", "Granger"],
                            Malfoy=["Malfoy", "Draco Malfoy", "Draco"], Hagrid=["Hagrid"], Snape=["Snape", "Severus"],
                            Quirrell=["Quirrell"], Dudley=["Dudley"], Neville=["Neville", "Neville Longbottom"],
                            McGonagall=["McGonagall"], Voldemort=["Voldemort"], Filch=["Filch"], Percy=["Percy"],
                            George=["George"], Fred=["Fred", "Fred Weasley"], Fluffy=["Fluffy"],
                            Goyle=["Goyle"], Norbert=["Norbert"], Charlie=["Charlie"], Crabbe=["Crabbe"],
                            Flitwick=["Flitwick"], Fang=["Fang"], Firenze=["Firenze"], Ollivander=["Ollivander"],
                            Bane=["Bane"], Seamus=["Seamus", "Seamus Finnigan"], Pomfrey=["Pomfrey"], Norris=["Norris"],
                            Hedwig=["Hedwig"], Hooch=["Hooch"], Lee=["Lee", "Lee Jordan"], Piers=["Piers"],
                            Dean=["Dean"], Flint=["Flint"], Figg=["Figg"], Malkin=["Malkin"], Scabbers=["Scabbers"],
                            Ronan=["Ronan"], Longbottom=["Longbottom"], Lily=["Lily"],
                            Ginny=["Ginny"], Marge=["Marge"], Trevor=["Trevor"], Patil=["Patil"],
                            Nicolas=["Nicolas", "Flamel"], James=["James", "James Potter"])

persons_list = sorted(list(person_dict.items()), key=lambda x: x[1], reverse=True)
paragraphes = one_paragraphe  # Rappel de la variable paragraphes

# Nettoyage des paragraphes
para_clean = []
for para in paragraphes:
    sentences = sent_detector.tokenize(para.strip())
    new_para = ""
    for sentence in sentences:
        sentence_clean = delete_punctuation(sentence)
        new_para = new_para + sentence_clean
    para_clean.append(new_para)


def text_tokenize(book):
    tokenize = nltk.word_tokenize(book)
    return tokenize


# Fonction qui retourne true si le perso est dans un paragraphe, false sinon
def present_in(noun, para):
    b = text_tokenize(para)
    for word in b:
        if noun == word:
            return True
    return False


def presence_in_hp(noun):
    presence = []
    counter = 0
    for para in para_clean:
        if present_in(noun, para):
            presence.append(counter)
        counter += 1
    return presence


# Traitement du texte pour préparer à Word2Vec
all_sentences_clean2 = []
for sent in tqdm(all_sentences_clean):
    sent_clean = sent
    for x, y in manual_character_dic.items():
        for k in y:
            sent_clean = sent_clean.replace(k, x)
    all_sentences_clean2.append(sent_clean)

CUSTOM_FILTERS = [lambda x: x.lower(), strip_non_alphanum, remove_stopwords, strip_multiple_whitespaces]

all_sentences_preprocessed = []
for sent in tqdm(all_sentences_clean2):
    parsed_line = preprocess_string(sent, CUSTOM_FILTERS)
    all_sentences_preprocessed.append(parsed_line)

# %% CO-OCCURENCE NETWORK + LOUVAIN


cara_list = []  # Crée une liste de list des personnages, les sous-listes comprennant le doublons des noms
for x in manual_character_dic.values():
    cara_list.append(x)


# Fonction qui combine 2 list en une seule liste
def combine_list(list1, list2):
    pos = len(list1)
    for i in range(len(list2)):
        list1.insert(i + pos, list2[i])
    return list1


# Creé une matric avec les personnages et leur occurences dans les paragraphes
occurences_persos = [1] * len(cara_list)
i = 0
for car in cara_list:
    occurences_perso = []
    for perso in car:
        combine_list(occurences_perso, presence_in_hp(perso))
    occurences_perso.sort()
    occurences_persos[i] = occurences_perso
    i += 1

# Crée un dictionnaire affectant les présences dans les paragraphes au persos
final_dic = {}
i = 0
for car in cara_list:
    final_dic[car[0]] = occurences_persos[i]
    i += 1


# Fonction qui prend 2 listes en arguments et qui retourne le noumbres d'éléments en communs
def common(list1, list2):
    com = [value for value in list1 if value in list2]
    return len(com)


# Création du netwok
G = nx.Graph()
for car in final_dic:
    G.add_node(car)

# Ajout des edges au network
copy_final_dic = final_dic.copy()
for car1 in final_dic:
    copy_final_dic.pop(car1)
    for car2 in copy_final_dic:
        link = common(final_dic[car1], copy_final_dic[car2])
        if link != 0:
            G.add_edge(car1, car2, weight=link)

nx.draw(G, with_labels=True, font_weight='bold')
plt.show()

Assort = nx.degree_assortativity_coefficient(G)
commu = community_louvain.best_partition(G)
# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(commu.values()) + 1)
nx.draw_networkx_nodes(G, pos, commu.keys(), label=True, node_size=40,
                       cmap=cmap, node_color=list(commu.values()))
nx.draw_networkx_edges(G, pos, alpha=0.5)
plt.show()
# %% WORD2VEC MODEL + K-MEANS

model = gensim.models.Word2Vec(size=100, window=4, min_count=1, alpha=0.05)  # DEFINE MODEL

# BUILD VOCABULARY
model.build_vocab(all_sentences_preprocessed)

# AND TRAIN THE MODEL
iterations = tqdm(range(10))
for i in iterations:
    model.train(all_sentences_preprocessed, total_examples=model.corpus_count, compute_loss=True,
                epochs=200)
    msg = f"Iter :: {i} -- Loss :: {model.get_latest_training_loss()}"
    iterations.set_postfix_str(s=msg, refresh=True)

# GET THE VOCABULARY FROM THE MODEL
vocabulary = list(model.wv.vocab)

# GET THE WORD EMBEDDING VECTORS
embedding_vectors = model[model.wv.vocab]

# GET THE LOWER CASE REPRESENTATION OF NAMED ENTITIES THAT ARE IN THE VOCABULARY
person_set_processed = [s for s in manual_character_dic.keys() if s.lower() in vocabulary]
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

# transforming the embedding using PCA
pca = PCA(n_components=2)
Y_transform = pca.fit(embedding_vectors).transform(embedding_vectors)

# PLOT THE PROJECTION
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*V_tranform), marker='.', s=50, lw=0, alpha=0.7,
            edgecolor='k')
# for i, (x,y) in enumerate(V_tranform):
#     plt.text(x,y, vocabulary[i],
#                 fontsize = 4, alpha = 0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig('Projection using TSNE.png')
plt.show()

fig2 = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*Y_transform), marker='.', s=50, lw=0, alpha=0.7,
            edgecolor='k')
# for i, (x,y) in enumerate(V_tranform):
#     plt.text(x,y, vocabulary[i],
#                 fontsize = 4, alpha = 0.5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig('Projection using PCA.png')
plt.show()

kpca = KernelPCA(n_components=2, kernel="sigmoid")
X_kpca = kpca.fit_transform(embedding_vectors)

fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*X_kpca), marker='.', s=50, lw=0, alpha=0.7,
            edgecolor='k')
# for i, (x,y) in enumerate(V_tranform):
#     plt.text(x,y, vocabulary[i],
#                 fontsize = 4, alpha = 0.5)
plt.xlabel("Pkernel-CA 1")
plt.ylabel("kernel-PCA 2")
plt.savefig('Projection using kernel-PCA.png')
plt.show()

# THIS IS THE IMPLEMENTATION OF THE KMEANS WITH COSINE DISTANCE

NUM_CLUSTERS = 5  # desired number of clusters to find

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
plt.savefig('projection using cosine clusters TSNE')

plt.show()

# Plot of the PCA results


fig4 = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*Y_transform), marker='.', s=50, lw=0, alpha=0.7, c=colors,
            edgecolor='k')

for i, (x, y) in enumerate(Y_transform):
    plt.text(x, y, vocabulary[i], color=colors[i],
             fontsize=10, alpha=0.5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig('projection using cosine clusters PCA')
plt.show()

# THIS IS THE IMPLEMENTATION OF THE KMEANS WITH EUCLIDEAN DISTANCE
from sklearn.cluster import KMeans

NUM_CLUSTERS = 5

kclusterer_sklearn = KMeans(n_clusters=NUM_CLUSTERS)

assigned_clusters = kclusterer_sklearn.fit_predict(V_tranform)
assigned_clusters2 = kclusterer_sklearn.fit_predict(Y_transform)

colors = cm.nipy_spectral(np.array(assigned_clusters).astype(float) / NUM_CLUSTERS)
colors2 = cm.nipy_spectral(np.array(assigned_clusters2).astype(float) / NUM_CLUSTERS)

# PLOT THE RESULTS
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*V_tranform), marker='.', s=50, lw=0, alpha=0.7, c=colors,
            edgecolor='k')

for i, (x, y) in enumerate(V_tranform):
    plt.text(x, y, vocabulary[i], color=colors[i],
             fontsize=4, alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig('projection using euclidean clusters TSNE')

plt.show()

fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*Y_transform), marker='.', s=50, lw=0, alpha=0.7, c=colors,
            edgecolor='k')

for i, (x, y) in enumerate(Y_transform):
    plt.text(x, y, vocabulary[i], color=colors[i],
             fontsize=4, alpha=0.5)
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig('projection using Euclidean clusters PCA')

plt.show()

# GET SUBSET OF VECTORS
person_embedding_vectors = model[person_set_vocab]

# PROJECT IT INTO 2D
A_tranform = TSNE(n_components=2).fit_transform(person_embedding_vectors)
assigned_clusters3 = kclusterer_sklearn.fit_predict(A_tranform)
colors3 = cm.nipy_spectral(np.array(assigned_clusters3).astype(float) / NUM_CLUSTERS)
B_tranform = pca.fit(person_embedding_vectors).transform(person_embedding_vectors)
assigned_clusters4 = kclusterer_sklearn.fit_predict(B_tranform)
clusters = list(commu.values())

# PLOT THE PROJECTION
fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*A_tranform), marker='.', s=100, lw=0, alpha=0.7,
            edgecolor='k')
for i, (x, y) in enumerate(A_tranform):
    plt.text(x, y, person_set_vocab[i], fontsize=12, alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig("character TSNE projection.png")
plt.show()

fig = plt.figure(figsize=(10, 8), dpi=90)
plt.scatter(*zip(*A_tranform), marker='.', s=50, lw=0, alpha=0.7, c=colors3,
            edgecolor='k')

for i, (x, y) in enumerate(A_tranform):
    plt.text(x, y, person_set_vocab[i], color=colors3[i],
             fontsize=12, alpha=0.5)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.savefig('projection of the clusters of characters using euclidean clusters TSNE')

plt.show()

word = "hermione"
similar_words = sorted(model.wv.most_similar(positive=[word], topn=10), key=lambda x: x[1], reverse=False)
X, Y = zip(*similar_words)
X = [x.upper() for x in X]
colors = plt.cm.jet(np.linspace(0, 1, len(X)))
fig = plt.figure(figsize=(3, 6), dpi=120)
plt.title(f"Most similar to :: {word.upper()}")
plt.barh(X, Y, align='center', height=0.75, color=colors)
plt.xlabel("Cosine similarity")
plt.show()


# %% jaccard similarity
def jaccard(labels1, labels2):
    """
    Computes the Jaccard similarity between two sets of clustering labels.
    The value returned is between 0 and 1, inclusively. A value of 1 indicates
    perfect agreement between two clustering algorithms, whereas a value of 0
    indicates no agreement. For details on the Jaccard index, see:
    http://en.wikipedia.org/wiki/Jaccard_index
    Example:
    labels1 = [1, 2, 2, 3]
    labels2 = [3, 4, 4, 4]
    print jaccard(labels1, labels2)
    @param labels1 iterable of cluster labels
    @param labels2 iterable of cluster labels
    @return the Jaccard similarity value
    """
    n11 = n10 = n01 = 0
    n = len(labels1)
    # TODO: Throw exception if len(labels1) != len(labels2)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)


jaccard(assigned_clusters4, clusters)