import os
import glob

from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

def getCorpus(corpus, path):
	print(path)
	for filename in glob.glob(os.path.join(path, '*.txt')):
		print(filename)
		print("prann")
		f = open(filename, 'r')
		txt = f.read()
		corpus.append(txt)
	return corpus


def main():
	vectorizer = CountVectorizer(stop_words='english')
	
	print("Reading positive reviews:")
	corpus_pos = []
	corpus_pos = getCorpus(corpus_pos, '../dataset/train/pos/')
	len_pos = len(corpus_pos)

	print("Reading negative reviews:")	
	corpus_tot = getCorpus(corpus_pos, '../dataset/train/neg/')
	len_neg = len(corpus_tot) - len_pos

	mat = (vectorizer.fit_transform(corpus_tot)).toarray()
	voc = vectorizer.vocabulary_
	voc_size = len(voc)
	mat = mat / (voc_size*1.0)
	print(mat.shape)

main()