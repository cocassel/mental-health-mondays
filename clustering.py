import pandas
import csv
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.cluster import KMeansClusterer
from nltk.stem import PorterStemmer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import itertools
from collections import OrderedDict

# intialize stemmer
stemmer = PorterStemmer()


# Function takes in data and builds the word2vec model
def createModel (data):
    comments = data['comments'].dropna().values.tolist()

    tokenized_stopped_corpus = []
    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenizing, removing stopwords, removing punctuation, lowercasing, and only keeping nouns
    for comment in comments:
        comment_words = tokenizer.tokenize(comment.decode('utf-8', errors='replace'))
        word_tags = nltk.pos_tag(comment_words)
        comment_words_stops_removed = []
        for word, pos in word_tags:
            if word.lower() not in stopwords.words('english'):
                if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                    comment_words_stops_removed.append(word.lower())

                    # We did not end up using stemming here. We tried it but it yielded extremely similar results

                    #stemmed_word = stemmer.stem(word.lower())
                    #post_words_stops_removed.append(stemmed_word)
        tokenized_stopped_corpus.append(comment_words_stops_removed)
    word2vec_model = gensim.models.Word2Vec(tokenized_stopped_corpus, min_count=1, size=200)

    # Get frequencies of words
    flat_list = list(itertools.chain(*tokenized_stopped_corpus))
    sum = 0
    word_dict = dict.fromkeys(flat_list, 0)
    for word in range(0,len(flat_list)):
        sum += 1
        for text, freq in word_dict.items():
            if flat_list[word] == text:
                word_dict[text] += 1

    ordered_dict = OrderedDict(sorted(word_dict.items(), key=lambda x: x[1]))
    print(ordered_dict)
    
    return word2vec_model


# Function takes in a word2vec model and creates clusters. Then it prints the clusters to CSV
def k_means_cluster(word2vec_model, num_clusters):
    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]
    #num_clusters = 100
    clusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.euclidean_distance, repeats=25)
    assigned_clusters = clusterer.cluster(X, assign_clusters=True)

    dict = {}
    for index in range(num_clusters):
        cluster_list = list()
        for i, word in enumerate(vocab):
            if assigned_clusters[i] == index:
                cluster_list.append(word)
        dict[index] = cluster_list

    with open("clusters/" + str(num_clusters) + 'clustering_result.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict.items():
            writer.writerow([key, value])


# Function takes in word2vec model and creates a scatterplot of all the words in a 2 dimensional space
def plot_scatterplot_all_words(word2vec_model):

    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    df_plot = pandas.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

    # scatter pyplot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df_plot['x'], df_plot['y'])

    for word, pos in df_plot.iterrows():
        ax.annotate(word, pos)

    plt.show()


# Function takes in a word2vec model and a word and creates a scatterplot of similar words to the word argument
def plot_scatterplot_closest_words(word2vec_model, word):

    vector_array = np.empty((0, 200), dtype='f')
    word_labels = [word]

    # get close words
    close_words = word2vec_model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    vector_array = np.append(vector_array, np.array([word2vec_model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = word2vec_model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        vector_array = np.append(vector_array, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(vector_array)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()


# Elbow Method for optimal number of clusters
def elbow_method(word2vec_model):

    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]
    distortions = []
    K = range(1, 20)
    for k in K:
        kmeanModel = KMeans(n_clusters=k * 25).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k / 25')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


# Plot the clusters in 2-dimensional space
def plot_clusters(word2vec_model):

    vocab = list(word2vec_model.wv.vocab)
    X = word2vec_model[vocab]

    kmeans = KMeans(n_clusters=250)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    plt.show()



# Read in mental health monday data predictions from csv
data = pandas.read_csv("predict_results_all.csv")

# Only use data that was deemed related by our prediction algorithm
data_predicted_related = data.loc[data['prediction'] == 'Related']
model = createModel(data_predicted_related)
plot_scatterplot_all_words(model)
elbow_method(model)
plot_clusters(model)
plot_scatterplot_closest_words(model, "interview")
plot_scatterplot_closest_words(model, "engineering")
plot_scatterplot_closest_words(model, "suicide")
plot_scatterplot_closest_words(model, "anxiety")
plot_scatterplot_closest_words(model, "depression")
# Print clusters for k = 25 to 475 in increments of 25
for i in range(1, 20):
    k_means_cluster(model, i*25)
