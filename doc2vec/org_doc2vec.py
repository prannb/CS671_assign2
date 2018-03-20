import gensim
import glob
import os
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from os import listdir
from os.path import isfile, join

corpus = []
doclabels = []

for filename in glob.glob(os.path.join('../pos/', '*.txt')):
		# print(filename)
		# print("prann")
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)

for filename in glob.glob(os.path.join('../neg/', '*.txt')):
		# print(filename)
		# print("prann")
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	
doclabels.append(1)
doclabels.append(0)

tokenizer = RegexpTokenizer(r'\w+')
stopword_set = set(stopwords.words('english'))
#This function does all cleaning of data using two objects above
def nlp_clean(data):
   new_data = []
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data


class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
              yield gensim.models.doc2vec.LabeledSentence(doc, [self.labels_list[idx]])

corpus = nlp_clean(corpus)
print corpus
exit()

it = LabeledLineSentence(corpus, doclabels)

model = gensim.models.Doc2Vec(size=300, min_count=0, alpha=0.025, min_alpha=0.025)
model.build_vocab(it)
#training of model
model.total_examples=model.corpus_count
# model.train(it)
for epoch in range(100):
	print 'iteration '+str(epoch+1)
	model.train(it)
	model.alpha -= 0.002
	model.min_alpha = model.alpha
