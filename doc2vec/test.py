from sklearn.feature_extraction.text import CountVectorizer
corpus = ["This is very strange stranger<br>/*-/*-",
          "This is very nice"]
vectorizer = CountVectorizer(stop_words='english')
# TfidfVectorizer(binary=True)
# print(vectorizer)
X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print dict(zip(vectorizer.get_feature_names(), idf))
print corpus[0]
print (vectorizer.vocabulary_)

from sklearn.feature_extraction.text import CountVectorizer
# corpus = ["This is very strange",
#           "This is very nice"]
vectorizer = CountVectorizer(stop_words='english')
# TfidfVectorizer(binary=True)
# print(vectorizer)
# X = vectorizer.fit_transform(corpus)
# idf = vectorizer.idf_
# print dict(zip(vectorizer.get_feature_names(), idf))
# print corpus[0]



#----------------------- first.py-----------------------------------------------
import os

import glob

yum =[]

path = '../dataset/train/pos'

for filename in glob.glob(os.path.join(path, '*.txt')):
	f = open(filename, 'r')
	corpus = f.read()
	yum.append(corpus)

print yum[1]
# y = vectorizer.build_preprocessor()
# y

X = vectorizer.fit_transform(yum)
X_array = (X.toarray())/55484.0
Y = X_array[0]
print(X_array)
for i in range(12400):
	Y = Y + X_array[i]
print(Y)
print len(vectorizer.vocabulary_)


# for i in X:
# 	print(i)

# for filename in os.listdir(path):
# 	print(filename)
# 	f = open(filename, 'r')
# 	corpus = f.read()
# 	yum.append([corpus])