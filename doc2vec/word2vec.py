import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=False)
# if you vector file is in binary format, change to binary=True
sentence = ["London", "is", "the", "capital", "of", "Great", "Britain"]
vectors = [model[w] for w in sentence]
print vectors