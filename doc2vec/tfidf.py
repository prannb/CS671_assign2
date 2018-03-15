import os
import glob

from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

def getCorpus(corpus, path):
	corpus = []
	for filename in tqdm(glob.glob(os.path.join(path, '*.txt'))):
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	return corpus


def main():
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
	# print(mat[0])
	# for i in mat[0]:
	# 	print i

main()