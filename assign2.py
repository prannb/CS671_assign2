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

nlp = spacy.load('en')

stop_words = set(stopwords.words('english'))

def getCorpus(corpus, path):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	return corpus

def bbow():
	vectorizer = CountVectorizer(stop_words='english', binary=True)
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, '../dataset/train/pos')
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, '../dataset/train/neg')
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_

def tf():
	vectorizer = CountVectorizer(stop_words='english')
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, '/dataset/train/pos')
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, '/dataset/train/neg')
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	voc_size = len(voc)
	mat = mat / (voc_size*1.0)
	print(mat.shape)

def tfidf():
	vectorizer = TfidfVectorizer(stop_words='english')
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, '../dataset/train/pos')
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, '../dataset/train/neg')
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	voc_size = len(voc)
	
	mat = np.array(mat)
	print(mat.shape)

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


def Glove():
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpusGlove(corpus_pos, '../dataset/train/pos')
	len_pos = len(corpus_pos)
	print(corpus_pos[0])

	print("Reading negative reviews:")	
	corpus_tot = getCorpusGlove(corpus_pos, '../dataset/train/neg')
	len_neg = len(corpus_tot) - len_pos