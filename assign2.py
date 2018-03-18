import os
import glob
import numpy as np
import re
import gensim
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from nltk.corpus import brown
from spacy.vectors import Vectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

#-------------- Document to Doc Vector --------------------------------------------------

nlp = spacy.load('en')

stop_words = set(stopwords.words('english'))

path_train_pos = 'dataset/train/pos/'
path_train_neg = 'dataset/train/neg/'
path_test_pos = 'dataset/test/pos/'
path_test_neg = 'dataset/test/neg/'

def randomise(mat, len_pos, len_neg):
	a = np.ones([len_pos, 1])
	b = np.zeros([len_neg, 1])
	c = np.concatenate((a,b), axis=0)
	mat_fin = np.concatenate((c,mat), axis=1)
	np.random.shuffle(mat_fin)
	return mat_fin

def getCorpus(corpus, path):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		# print(filename)
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	return corpus

def bbow(path1, path2):
	print("Document to Binary Bag of Words:")
	vectorizer = CountVectorizer(stop_words='english', binary=True)
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	mat = np.array(mat)
	return randomise(mat, len_pos, len_neg)

def tf(path1, path2):
	print("Document to tf matrix:")
	vectorizer = CountVectorizer(stop_words='english')
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	voc_size = len(voc)
	mat = mat / (voc_size*1.0)
	mat = np.array(mat)	
	# print(mat.shape)
	return randomise(mat, len_pos, len_neg)

def tfidf(path1, path2):
	print("Document to tfidf matrix:")
	vectorizer = TfidfVectorizer(stop_words='english')
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	voc_size = len(voc)
	
	mat = np.array(mat)
	# print(mat.shape)
	return randomise(mat, len_pos, len_neg)

def getGloveVec(txt):
	vec = 0
	num = 0
	word_tokens = word_tokenize(txt)
	for w in word_tokens:
		if (w not in stop_words) and re.match(r"\w", w):
			u_word = unicode(w)
			final = nlp(u_word)
			vec = vec + final.vector
			num = num + 1
	return (vec/(num*1.0))

def getCorpusGlove(corpus, path):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		vec = getGloveVec(txt)
		corpus.append(vec)
	return corpus


def glove(path1, path2):
	print("Document to average of Glove vectors:")
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpusGlove(corpus_pos, path1)
	len_pos = len(corpus_pos)
	# print(corpus_pos[0])

	print("Reading negative reviews:")	
	corpus_tot = getCorpusGlove(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos
	mat = np.array(corpus_tot)
	return randomise(mat, len_pos, len_neg)


#-------------- Classification Algorithms --------------------------------------------------

from sklearn.naive_bayes import GaussianNB
def naivebayes(xtrain, ytrain, xtest, ytest):
	gnb = GaussianNB()
	print("Performing Naive Bayes classification:")
	y_pred = gnb.fit(xtrain,ytrain).predict(xtest)
	# (iris.data.shape[0],(iris.target != y_pred).sum()))
	return y_pred

from sklearn.linear_model import LogisticRegression
# from sklearn.cross_validation import train_test_split
def logisticRegression(xtrain, ytrain, xtest, ytest):
	print("Performing Logistic Regression classification:")
	model = LogisticRegression()
	model = model.fit(xtrain, ytrain)
	return model.score(xtest, ytest)

from sklearn import svm
def supportVM(xtrain, ytrain, xtest, ytest):
	print("Performing SVM classification:")
	clf = svm.SVC()
	clf.fit(xtrain, ytrain)
	y_pred = clf.predict(xtest)
	return y_pred

from sklearn.neural_network import MLPClassifier
def neural_network(xtrain, ytrain, xtest, ytest):
	print("Performing Neural Network classification:")
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
	clf.fit(xtrain, ytrain)
	y_pred = clf.predict(xtest)
	return y_pred

#----------------- Combining the above two ---------------------------------------------------

def main():
	mat1 = tf(path_train_pos, path_train_neg)
	mat2 = tfidf(path_train_pos, path_train_neg)
	mat3 = bbow(path_train_pos, path_train_neg)
	# mat4 = glove(path_train_pos, path_train_neg)
	print(mat1.shape)
	print(mat2.shape)
	print(mat3.shape)
	# print(mat4.shape)

main()