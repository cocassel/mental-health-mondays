import json as j
import pandas
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import sys


################################################## Build the model ###################################################

home_improvement_comments = pandas.read_csv("homeimprovement_40000comments.csv")
home_improvement_comments['classification'] = 'Unrelated'
print("Home Improvement Comments: " + str(len(home_improvement_comments)))

mental_health_comments = pandas.read_csv("mentalhealth_40000comments_reduced.csv")
mental_health_comments['classification'] = 'Related'
print("Mental Health Comments: " + str(len(mental_health_comments)))

df_all_comments = pandas.concat([home_improvement_comments, mental_health_comments])

stemmer = SnowballStemmer('english')
stop_words = stopwords.words("english")

df_all_comments['cleaned'] = df_all_comments['comments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())


X_train, X_test, y_train, y_test = train_test_split(df_all_comments['cleaned'], df_all_comments.classification, test_size=0.2)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=1000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=4000, dual=False))])


model = pipeline.fit(X_train, y_train)

print("accuracy score: " + str(model.score(X_test, y_test)))




######################################################### Use the model ##########################################

data = pandas.read_csv("MHM_all_comments.csv")



### NOT STEMMING ###

comments = data['comments'].dropna().values.tolist()
classifications = data['classification'].dropna().values.tolist()

stripped_comments = []
for i in comments:
    stripped_comments.append(i.decode(encoding='UTF-8', errors='replace').strip())
# stripped_comments = filter(None, stripped_comments) # fastest

comments_nouns_stemmed = []
comments_nouns_stemmed_orig = []
classifications_nouns_stemmed = []

comments_nouns = []
comments_nouns_orig = []
classifications_nouns = []

comments_above_10_words = []
classifications_above_10_words = []


for comment, classification in zip(stripped_comments, classifications):

    comment_words = comment.split()
    # Weed out really short comments -- these are generally just agreement/disagreement to other comments
    if len(comment_words) > 10:
        word_tags = nltk.pos_tag(comment_words)
        comment_words_nouns = []
        comment_words_nouns_stemmed = []
        for word, pos in word_tags:
            if pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS':
                if word not in stop_words:
                    comment_words_nouns.append(word.lower())
                    comment_words_nouns_stemmed.append(stemmer.stem(word.lower()))
        if len(comment_words_nouns_stemmed) > 0:
            comments_nouns_stemmed.append(" ".join(comment_words_nouns_stemmed))
            comments_nouns_stemmed_orig.append(comment)
            classifications_nouns_stemmed.append(classification)
        if len(comment_words_nouns) > 0:
            comments_nouns.append(" ".join(comment_words_nouns))
            comments_nouns_orig.append(comment)
            classifications_nouns.append(classification)
        comments_above_10_words.append(comment)
        classifications_above_10_words.append(classification)







predictions = model.predict(comments_above_10_words)
predict_df = pandas.DataFrame(
    {'comments_orig': comments_above_10_words,
     'prediction': predictions,
     'classification': classifications_above_10_words
    })
predict_df.to_csv('predict_results.csv', index=False, encoding='utf-8')


predictions = model.predict(comments_nouns)
predict_df = pandas.DataFrame(
    {'comments_orig': comments_nouns_orig,
     'comments_nouns': comments_nouns,
     'prediction': predictions,
     'classification': classifications_nouns
    })
predict_df.to_csv('predict_results_nouns.csv', index=False, encoding='utf-8')


predictions = model.predict(comments_nouns_stemmed)
predict_df = pandas.DataFrame(
    {'comments_orig': comments_nouns_stemmed_orig,
     'comments_nouns_stemmed': comments_nouns_stemmed,
     'prediction': predictions,
     'classification': classifications_nouns_stemmed
    })
predict_df.to_csv('predict_results_nouns_stemmed.csv', index=False, encoding='utf-8')




comments_above_10_words = []

for comment in stripped_comments:
    comment_words = comment.split()
    # Weed out really short comments -- these are generally just agreement/disagreement to other comments
    if len(comment_words) > 10:
        comments_above_10_words.append(comment)

predictions = model.predict(comments_above_10_words)
predict_df = pandas.DataFrame(
    {'comments': comments_above_10_words,
     'prediction': predictions,
    })
predict_df.to_csv('predict_results_all.csv', index=False, encoding='utf-8')



#data = pandas.read_csv("MHM_all_comments.csv")

#data['cleaned'] = data['comments'].apply(lambda x: " ".join([i for i in re.sub("[^a-zA-Z]", " ", x).split()]).lower())

# comments = data['cleaned'].dropna().values.tolist()








