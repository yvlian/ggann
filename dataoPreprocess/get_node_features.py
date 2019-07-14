import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy import sparse

# Token
tknzr = TweetTokenizer()
# Stop Words
stopwords = nltk.corpus.stopwords.words("english")
stopwords = set(stopwords)
# Stemming
english_stemmer = nltk.stem.PorterStemmer()
#Tf-idf Vectorize
vectorizer = TfidfVectorizer()
problem = pd.read_csv('selected_problem')
problem.fillna(' ',inplace=True)
problem['description'] = (problem['description'] +' '+ problem['input'] +' '+ problem['output']).apply(
    lambda s:re.sub(r'[^a-zA-Z0-9<=>\+\-/\s]',repl=' ',string=s))
problem['description'] = problem['description'].apply(tknzr.tokenize)
problem['description'] = problem['description'].apply(lambda x:' '.join([english_stemmer.stem(s.lower()) for s in x]))
vector = vectorizer.fit_transform(problem['description'].values)
X = vector.toarray()
pca = PCA(n_components=29)
newX = pca.fit_transform(X)
to_save = dict()
for i in range(problem.shape[0]):
    to_save[str(problem['id'][i])] = X[i].reshape(1,-1).tolist()
js = json.dumps(to_save)
with open('tf_idf_vector.txt','w') as f:
    f.write(js)
# print(pca.explained_variance_ratio_)
# problem.to_csv('tf_idf_vector.txt',index=False)
a = 1