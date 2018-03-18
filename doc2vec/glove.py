import spacy
import os
import glob
import numpy as np
import re

from spacy.vectors import Vectors
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

nlp = spacy.load('en')

stop_words = set(stopwords.words('english'))

def getGloveVec(txt, filename):
	vec = 0
	num = 0
	word_tokens = word_tokenize(txt)
	for w in word_tokens:
		if (w not in stop_words) and re.match(r"\w", w):
			print(w + " " + filename)
			try:
				u_word = unicode(w)
			except UnicodeDecodeError:
				continue
			final = nlp(u_word)
			vec = vec + final.vector
			num = num + 1
	return (vec/(num*1.0))

def getCorpusGlove(corpus, path):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		vec = getGloveVec(txt, filename)
		corpus.append(vec)
	return corpus


def main():
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpusGlove(corpus_pos, '../dataset/train/pos')
	len_pos = len(corpus_pos)
	print(corpus_pos[0])

	print("Reading negative reviews:")	
	corpus_tot = getCorpusGlove(corpus_pos, '../dataset/train/neg')
	len_neg = len(corpus_tot) - len_pos

	for i in corpus[0]:
		print i

main()

# nlp = spacy.load('en')
# # vector_table = numpy.zeros((3, 300), dtype='f')
# # doc = Vectors([u'human'], vector_table)

# # doc1 = nlp(u'prann')
# # doc2 = nlp(u'best')
# a = "prann asdsa"
# ua = unicode(a)
# doc3 = nlp(ua)
# print (doc3.vector)
# # print (doc1.vector + doc2.vector)