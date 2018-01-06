import pandas as pd

if __name__ == '__main__':
    df_train = pd.read_csv('./train.csv', header=None, sep='\t', names=['source', 'title'])

    inputs_train = df_train['title']
    classes_train = df_train['source']

    print(inputs_train[0])

import nltk

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

from sklearn.feature_extraction.text import CountVectorizer

# class StemmedCountVectorizer(CountVectorizer):
#     def build_analyzer(self):
#         analyzer = super(StemmedCountVectorizer, self).build_analyzer()
#         return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

#stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
if __name__ == '__main__':
    #parameters = {'vect__ngram_range': [(1, 1), (1, 2)],'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3),}

    #text_clf_svm = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),('clf', MultinomialNB(fit_prior=False))])
    #text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()),('clf', MultinomialNB(fit_prior=False))])
    text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])

    #gs_clf = GridSearchCV(text_clf_svm, parameters, n_jobs=-1)
    #gs_clf = gs_clf.fit(inputs_train, classes_train)
    text_clf_svm = text_clf_svm.fit(inputs_train, classes_train)

    from sklearn.externals import joblib
    joblib.dump(text_clf_svm, './text_clf_svm_model.pkl') 

    df_eval = pd.read_csv('./eval.csv', header=None, sep='\t', names=['source', 'title'])

    inputs_eval = df_eval['title']
    classes_eval = df_eval['source']

import numpy as np
# predicted_svm = text_clf_svm.predict(inputs_eval)
if __name__ == '__main__':
    predicted_svm = text_clf_svm.predict(inputs_eval)
    print(np.mean(predicted_svm == classes_eval))


