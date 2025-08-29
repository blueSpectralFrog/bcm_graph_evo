"""
The point of this file is to create a set of "nodes" which have interests in 
various topics. Sentences from the OCEAN-synthetic dataset (MTHR/OCEAN on huggingface)
were put through an LdaModel to get num_topics topics. The sentences were then 
classified by probability for each topic. Saved to a csv. 
"""
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')

from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import numpy as np
import pandas as pd

def parse_topic_output(topic_output):
    """
    parses word and probability for a topic returned from LdaModel
    """
    topic_data = []
    probWord = topic_output.split(' + ')
    
    pattern = re.compile('[\W_]+')

    for pair in probWord:
        pair = pair.split('*')
        word = pattern.sub('', pair[1])
        prob = float(pair[0])
        topic_data.append({'word': word, 'probability': prob})
    return topic_data

def classify_sentence_by_topic(sentence, topicDist):
    """
    oh boy do I love nested loops in python
    Obv, the functions does not take into account context 
    it's just a classifier based on words/probas from the LdaModel
    """
    topicProb = np.zeros(len(topicDist))
    for word in sentence:
        for idx, topic in enumerate(topicDist):
            for subdict in topic:
                if word.lower() == subdict['word']:
                    topicProb[idx] += subdict['probability']

    return topicProb

dataset = pd.read_csv(r'C:\Users\ndnde\Documents\Projects\ML\datasets\OCEAN attributes\OCEAN-synthetic.csv')
sentences = dataset.iloc[:,0].tolist()

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

processed_docs = []
for doc in sentences:
    tokens = word_tokenize(doc.lower())
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(filtered_tokens)

dictionary = corpora.Dictionary(processed_docs)
corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

num_topics = 10
lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

topicDist = []
for idx, topic in lda_model.print_topics(-1):
    topicDist.append(parse_topic_output(topic))

topicProb = []
count = 0
for idx, sentence in enumerate(processed_docs):
    topicProb.append(classify_sentence_by_topic(sentence, topicDist))

    if sum(topicProb[-1]) == 0:
        # adjust topic number hyperparameter 
        count = count + 1
        
        # all equal probability. Unsure if need to normalize all the other 
        # sentences in this case... so for now I'll just set all probabilites 
        # to a tenth of the uniform distribution 
        topicProb[-1] = [1/num_topics/10]*num_topics
        
print(count)

for idx in range(num_topics):
    dataset['Topic_'+str(idx+1)] = [p[idx] for p in topicProb]

dataset.to_csv(r'C:\Users\ndnde\Documents\Projects\ML\datasets\OCEAN attributes\OCEAN-synthetic_topics.csv')