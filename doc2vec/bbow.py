import os
import glob

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def getCorpus(corpus, path):
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	return corpus


def main():
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
	# # print(mat_neg)
	# Y = mat_neg[0]
	# for i in range(12400):
	# 	Y = Y + mat_neg[i]
	# # print(Y)
	# for i in range(53000):
	# 	print Y[i]
	

main()