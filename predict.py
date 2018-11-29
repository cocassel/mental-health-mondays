import json as j
import pandas
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
import sys

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

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 3), stop_words="english", sublinear_tf=True)),
                     ('chi',  SelectKBest(chi2, k=5000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])


model = pipeline.fit(X_train, y_train)


# vectorizer = model.named_steps['vect']
# chi = model.named_steps['chi']
# clf = model.named_steps['clf']
#
# feature_names = vectorizer.get_feature_names()
# feature_names = [feature_names[i] for i in chi.get_support(indices=True)]
# feature_names = np.asarray(feature_names)
#
# target_names = ['Unrelated', 'Related']

# print("top 10 keywords per class:")
# for i, label in enumerate(target_names):
#     top10 = np.argsort(clf.coef_[i])[-10:]
#     print("%s: %s" % (label, " ".join(feature_names[top10])))
#
#
# top10 = np.argsort(clf.coef_[0])[-10:]
# print("%s: %s" % (label, " ".join(feature_names[top10])))
# top10 = np.argsort(clf.coef_[1])[-10:]
# print("%s: %s" % (label, " ".join(feature_names[top10])))
#

# print(model.predict(['that was an awesome place. Great food!']))
# print(model.predict(['I really hate life!']))


print("accuracy score: " + str(model.score(X_test, y_test)))


data = pandas.read_csv("MHM_all_comments.csv")

comments = data['comments'].dropna().values.tolist()

data['cleaned'] = data['comments'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in stop_words]).lower())

comments_stemmed = data['cleaned'].dropna().values.tolist()

stripped_comments = []
for i in comments:
    stripped_comments.append(i.decode(encoding='UTF-8', errors='replace').strip())



# Not stemming the input
predictions = model.predict(stripped_comments)

predict_df = pandas.DataFrame(
    {'comments': stripped_comments,
     'prediction': predictions
    })
predict_df.to_csv('predict_results.csv', index=False, encoding='utf-8')



# Stemming the input
predictions = model.predict(comments_stemmed)
predict_df = pandas.DataFrame(
    {'comments': comments_stemmed,
     'prediction': predictions
    })

predict_df.to_csv('predict_results_stemmed', index=False, encoding='utf-8')
