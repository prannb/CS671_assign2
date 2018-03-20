import os
import glob
import numpy as np
import re
import gensim
import spacy
import nltk

from scipy import sparse
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import normalize
from tqdm import tqdm
# from nltk.corpus import brown
from spacy.vectors import Vectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from gensim.models.keyedvectors import KeyedVectors

#-------------- Document to Doc Vector --------------------------------------------------

baseline = 5

nlp = spacy.load('en')

stop_words = set(stopwords.words('english'))

path_train_feat = 'dataset/train/rand_labeledBow.feat'
path_test_feat = 'dataset/test/rand_labeledBow.feat'

path_train_pos = 'dataset/train/pos/'
path_train_neg = 'dataset/train/neg/'
path_test_pos = 'dataset/test/pos/'
path_test_neg = 'dataset/test/neg/'
path_unsup = '../dataset/train/unsup'

def randomise(corpus_tot, len_pos, len_neg):
	s = np.arange(len(corpus_tot))
	np.random.shuffle(s)
	a = np.ones([len_pos])
	b = np.zeros([len_neg])
	c = np.concatenate((a,b))
	d = c
	corpus_rand = []
	count = 0
	for i in s:
		corpus_rand.append(corpus_tot[i])
		d[count] = c[i]
		count = count + 1
		# print count
	return (corpus_rand, d)
	

def getCorpus(corpus, path):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		# print(filename)
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	return corpus

def getVocab(path1, path2):
	print("Getting Vocab:")
	vectorizer = CountVectorizer(stop_words='english', binary=True)
	
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	# mat = np.array(mat)
	return voc

def bbow(path1, path2, voc):
	print("Document to Binary Bag of Words:")
	
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos

	corpus_tot = np.array(corpus_tot)
	print len(corpus_tot[0])
	corpus_fin = randomise(corpus_tot, len_pos, len_neg)

	corpus_tot = corpus_fin[:, 1:]
	y = corpus_fin[:, 0:1]

	(corpus_rand, y) = randomise(corpus_tot, len_pos, len_neg)
	mat = (vectorizer.fit_transform(corpus_tot))
	return mat

def tf(which, path1, path2, voc):
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos
	(corpus_rand, y) = randomise(corpus_tot, len_pos, len_neg)

	if (which == 'tf'):
		print("Document to tf matrix:")
		vectorizer = CountVectorizer(stop_words='english', vocabulary=voc)
		mat = vectorizer.fit_transform(corpus_rand)
		voc_size = len(voc)
		mat = mat / (voc_size*1.0)
	elif (which == 'tfidf'):
		print("Document to tfidf matrix:")
		vectorizer = TfidfVectorizer(stop_words='english', vocabulary=voc)
		mat = (vectorizer.fit_transform(corpus_rand)).toarray()
	else:
		print("Document to bbow matrix:")
		vectorizer = CountVectorizer(stop_words='english', binary=True, vocabulary=voc)
		mat = vectorizer.fit_transform(corpus_rand)
	return (mat, y)

def feat_tf(which, path, voc):
	data = load_svmlight_file(path)
	xdata = data[0]
	ydata = data[1]
	ydata = (ydata > baseline)
	if (which == 'tf'):
		print("Document to tf matrix:")
		xdata = normalize(xdata, norm='l1', axis=1)
	elif (which == 'tfidf'):
		print("Document to tfidf matrix:")
		tfidf = TfidfTransformer()
		xdata = tfidf.fit_transform(xdata)
	else:
		print("Document to bbow matrix:")
		xdata = (xdata != 0)
	return (xdata, np.array(ydata))

def tfidf(path1, path2):
	print("Document to tfidf matrix:")
	
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, path1)
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, path2)
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc_size = len(voc)
	
	mat = np.array(mat)
	# print(mat.shape)
	return randomise(mat, len_pos, len_neg)


def getGloveVec(txt, voc, filename):
	vec = 0
	num = 0
	word_tokens = txt.split(' ')
	for w in word_tokens:
		if re.match(r"\w+", w):
			to = re.match(r"\w+", w)
			u_word = unicode(to.group)
			final = nlp(u_word)
			vec = vec + final.vector
			num = num + 1
	return (vec/(num*1.0))

def getCorpusGlove(corpus, path, voc):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		vec = getGloveVec(txt, voc, filename)
		corpus.append(vec)
	return corpus

def glove(path1, path2, voc):
	print("Document to average of Glove vectors:")
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpusGlove(corpus_pos, path1, voc)
	len_pos = len(corpus_pos)
	# print(corpus_pos[0])

	print("Reading negative reviews:")	
	corpus_tot = getCorpusGlove(corpus_pos, path2, voc)
	len_neg = len(corpus_tot) - len_pos
	mat = np.array(corpus_tot)
	(corpus_rand, y) = randomise(mat, len_pos, len_neg)
	return (np.array(corpus_rand), y)

def getWordVec(txt, voc, filename, model):
	vec = 0
	num = 0
	word_tokens = txt.split(' ')
	for w in word_tokens:
		if re.match(r"\w+", w):
			to = re.match(r"\w+", w)
			try:
				final = model[to.group()]
			except KeyError:
				continue
			vec = vec + final
			num = num + 1
	return (vec/(num*1.0))

def getCorpusWord(corpus, path, voc, model):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		vec = getWordVec(txt, voc, filename, model)
		corpus.append(vec)
	return corpus

def word2vec(path1, path2, voc):
	print("Document to average of Word2Vec vectors:")
	import gensim
	from nltk.corpus import brown
	sentences = brown.sents()
	model = gensim.models.Word2Vec(sentences, min_count=1)
	model.save('brown_model')
	model = gensim.models.Word2Vec.load('brown_model')
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpusWord(corpus_pos, path1, voc, model)
	len_pos = len(corpus_pos)
	# print(corpus_pos[0])

	print("Reading negative reviews:")	
	corpus_tot = getCorpusWord(corpus_pos, path2, voc, model)
	len_neg = len(corpus_tot) - len_pos
	mat = np.array(corpus_tot)
	(corpus_rand, y) = randomise(mat, len_pos, len_neg)
	return (np.array(corpus_rand), y)

def gensim_getCorpusWord(path1, voc):
	data = load_svmlight_file(path)
	xdata = data[0]
	ydata = data[1]
	ydata = (ydata > baseline)
	model = KeyedVectors.load_word2vec_format('../GoogleNews.bin', binary=True)
	corpus = []
	vec = 0
	num = 0
	for i in xdata:
		for y in i.nonzero()[1]:
			w = voc[y]
			w = w.split("\n")
			w = w[0]
			try:
				final = model[w]
			except KeyError:
				continue
			vec = vec + final
			num = num + 1
			# print w
		vec = (vec/(num*1.0))
		# print num
		corpus.append(vec)
	return (np.array(corpus), np.array(ydata))

def gensim_getCorpusGlove(path, voc):
	print("Document to average of Glove vectors:")
	data = load_svmlight_file(path)
	xdata = data[0]
	ydata = data[1]
	ydata = (ydata > baseline)
	model = KeyedVectors.load_word2vec_format('../glove.6B/glove.6B.300d.txt', binary=False)
	corpus = []
	vec = 0
	num = 0
	for i in xdata:
		for y in i.nonzero()[1]:
			w = voc[y]
			w = w.split("\n")
			w = w[0]
			try:
				final = model[w]
			except KeyError:
				continue
			vec = vec + final
			num = num + 1
			# print w
		# break
		vec = (vec/(num*1.0))
		# print num
		corpus.append(vec)
	return (np.array(corpus), np.array(ydata))

def read_doc2vec(path, documents, tag):
	tokenizer = nltk.RegexpTokenizer(r'\w+')
	for filename in glob.glob(os.path.join('../dataset/train/pos/', '*.txt')):
		f = open(filename, 'r')
		txt = f.read()
		words = tokenizer.tokenize(txt)
		tags = [tag]
		tag = tag + 1
		documents.append(gensim.models.doc2vec.TaggedDocument(words=words,tags=tags))
	return (documents, tag)

def doc2vec(path1, path2, path3, path4, path5):
	documents = []
	(documents, tag_pos) = read_doc2vec(path1, documents, 0)
	(documents, tag_neg) = read_doc2vec(path2, documents, tag_pos)
	(documents, tag_pos_t) = read_doc2vec(path3, documents, tag_neg)
	(documents, tag_neg_t) = read_doc2vec(path4, documents, tag_pos_t)
	# (documents, tag) = read_doc2vec(path5, documents, tag_neg_t)
	model = gensim.models.Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
	model.train(documents, total_examples=len(documents), epochs=10)
	# model.save('my_model.doc2vec')
	corpus = []
	# ytrain = np.zeros([tag_neg])
	# ytest = np.zeros([tag_neg_t - tag_neg])
	for i in range(tag_neg_t):
		corpus.append(model[i])
		# if (i<tag_pos):
		# 	ytrain[i] = 1
		# if (i>=tag_neg) and (i<tag_pos_t):
		# 	ytest[i] = 1
	corpus = np.array(corpus)
	xtrain = corpus[0:tag_neg]
	xtest = corpus[tag_neg:]
	(xtrain, ytrain) = randomise(xtrain, tag_pos, tag_neg-tag_pos)
	(xtest, ytest) = randomise(xtest, tag_pos_t - tag_neg, tag_neg_t-tag_pos_t)
	return (xtrain, ytrain, xtest, ytest)

	
#-------------- Classification Algorithms --------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
def lstm(path1, path2, voc):
	model = KeyedVectors.load_word2vec_format('../glove.6B/glove.6B.50d.txt', binary=False)
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpusWord(corpus_pos, path1, voc, model)
	len_pos = len(corpus_pos)
	# print(corpus_pos[0])

	print("Reading negative reviews:")	
	corpus_tot = getCorpusWord(corpus_pos, path2, voc, model)
	len_neg = len(corpus_tot) - len_pos
	mat = np.array(corpus_tot)
	(corpus_rand, y) = randomise(mat, len_pos, len_neg)

	# max_review_length = 500
	some = np.array(corpus_rand)
	print some
	print some.shape
	exit()
	xdata = sequence.pad_sequences(corpus_rand, maxlen=max_review_length)
	# print xdata
	print xdata.shape
	# return
	embedding_vecor_length = 32
	model = Sequential()
	model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
	model.add(LSTM(100))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	model.fit(xdata, y, epochs=3, batch_size=64)	

def getAccuracy(ytest, y_pred):
	tot = 0.0
	for i in range(ytest.shape[0]):
		if ytest[i] == y_pred[i]:
			tot = tot + 1
	return tot/(ytest.shape[0] * 1.0)

from sklearn.naive_bayes import MultinomialNB
def naivebayes(xtrain, ytrain, xtest, ytest):
	gnb = MultinomialNB()
	print("Performing Naive Bayes classification:")
	y_pred = gnb.fit(xtrain,ytrain).predict(xtest)
	# (iris.data.shape[0],(iris.target != y_pred).sum()))
	accur = getAccuracy(ytest, y_pred)
	return accur

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
	accur = getAccuracy(ytest, y_pred)
	return accur

from sklearn.neural_network import MLPClassifier
def neural_network(xtrain, ytrain, xtest, ytest):
	print("Performing Neural Network classification:")
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(15, 10), random_state=1)
	clf.fit(xtrain, ytrain)
	y_pred = clf.predict(xtest)
	accur = getAccuracy(ytest, y_pred)
	return accur

#----------------- Combining the above two ---------------------------------------------------

def main():
	# voc = getVocab(path_train_pos, path_train_neg)
	voc = {}
	with open("dataset/imdb.vocab") as f:
		i = 0
		for line in f:
			val = line
			voc[i] = val
			i = i + 1
	
	# (xtrain, ytrain) = gensim_getCorpusGlove(path_train_feat, voc)
	# print(xtrain.shape)
	# (xtest, ytest) = gensim_getCorpusWord(path_test_feat, voc)
	# (xtrain,ytrain) = word2vec(path_train_pos, path_train_neg, voc)
	# print(xtrain.shape)
	# (xtest,ytest) = word2vec(path_test_pos, path_test_neg, voc)
	 
	# exit()
	# ytrain = mat1[:, 0:1]
	# ytrain = ytrain.reshape((ytrain.shape[0],))	
	# xtrain = mat1[:, 1:]
	# (xtest, ytest) = tf('bbow', path_test_pos, path_test_neg, voc)
	# ytest = mat1[:, 0:1]
	# ytest = ytest.reshape((ytest.shape[0],))
	# print ytest.shape
	# ytest.reshape(2012)
	# xtest = mat1[:, 1:]
	# print neural_network(xtrain, ytrain, xtest, ytest)
	# print naivebayes(xtrain, ytrain, xtest, ytest)
	# print xtrain
	# print ytrain
	# ytrain = sparse.csr_matrix(ytrain)
	# print ytrain
	# lstm(path_train_pos, path_train_neg, voc)
	# exit()
	# (xtrain, ytrain) = gensim_getCorpusGlove(path_train_feat, voc)
	# exit()
	# (xtest, ytest) = gensim_getCorpusGlove(path_test_feat, voc)
	# (xtrain, ytrain) = feat_tf('tf', path_train_feat, voc)
	# (xtest, ytest) = feat_tf('tf', path_test_feat, voc)
	# padding = np.zeros((xtest.shape[0],4))
	# xtest = sparse.hstack((xtest,padding))
	(xtrain, ytrain, xtest, ytest) = doc2vec(path_train_pos, path_train_neg, path_test_pos, path_test_neg, path_unsup)
	print logisticRegression(xtrain, ytrain, xtest, ytest)	
	# print supportVM(xtrain, ytrain, xtest, ytest)
	

main()