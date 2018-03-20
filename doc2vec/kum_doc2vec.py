import gensim
import glob
import os
import nltk
from gensim.models.doc2vec import TaggedDocument
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join
import numpy as np

tokenizer = nltk.RegexpTokenizer(r'\w+')
def nlp_clean(data):
	dlist = tokenizer.tokenize(data)
	return dlist
tag = 0
documents = []
for filename in glob.glob(os.path.join('../dataset/train/pos/', '*.txt')):
		# print(filename)
		# print("prann")
	f = open(filename, 'r')
	txt = f.read()
	# data = normalize_text(txt)
	words = nlp_clean(txt)
	tags = [tag]
	documents.append(TaggedDocument(words=words,tags=tags))
	tag = tag + 1
	# corpus.append(txt)
# np.save('documents.npy', documents)
print(" 1 done")


for filename in glob.glob(os.path.join('../dataset/train/pos/', '*.txt')):
		# print(filename)
		# print("prann")
	f = open(filename, 'r')
	txt = f.read()
	# data = normalize_text(txt)
	words = nlp_clean(txt)
	tags = [tag]
	tag = tag + 1
	documents.append(TaggedDocument(words=words,tags=tags))
	# corpus.append(txt)
# np.save('documents.npy', documents)
print(" 2 done")

# i=0
# for file in negtsfiles:
# 	os.system("perl -pi -e 's/[^[:ascii:]]//g' {}".format(file))
# 	with open(file, 'r') as curfile:
# 		data = curfile.read().decode("utf-8")
# 		data = normalize_text(data)
# 		words = nltk.word_tokenize(data)
# 		tags = [file]
# 		documents.append(TaggedDocument(words=words,tags=tags))
# 	print "  Iteration %d\r" % (i) ,
# 	i+=1
# np.save('documents.npy', documents)
# print("4 done")
# print documents
model = gensim.models.Doc2Vec(documents, vector_size=300, window=5, min_count=1, workers=4)
# model.build_vocab(documents)
model.train(documents, total_examples=len(documents), epochs=10)
# print(model['Sent_2'])
print("Trained")
model.save('my_model.doc2vec')
# load the model back
model_loaded = gensim.models.Doc2Vec.load('my_model.doc2vec')
print model_loaded[1]