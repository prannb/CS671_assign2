from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
example_sent = "This is a sample sentence, showing off the stop words br filtration."
 
stop_words = set(stopwords.words('english'))
 
word_tokens = word_tokenize(example_sent)
 
# filtered_sentence = [w for w in word_tokens if not w in stop_words]
 
# filtered_sentence = []
 
for w in word_tokens:
    if (w not in stop_words) and re.match(r"\w", w):
        # filtered_sentence.append(w)
		print w
# print(word_tokens)
# print(filtered_sentence)
