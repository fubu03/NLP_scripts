# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:44:05 2019

@author: sachin.kalra
"""

txt=open(r'C:\Users\sachin.kalra\Desktop\nlp_ss\text_file.txt','r').readlines()
txt1=''.join(txt)

from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
                ##########################
                ####text preprocessing:###
                ##########################

#1.lower case, stopwords, only alphabets

tokens=[w.lower() for w in word_tokenize(txt1) if w.isalpha()]
no_stop=[t for t in tokens if t not in stopwords.words('english')]

len(tokens)-len(no_stop)

Counter(no_stop)

#2. Lemmatization

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

lemma=[lemmatizer.lemmatize(t) for t in no_stop]

x,y=Counter(lemma).most_common(15)
                #######################
                #### Gensim Library####
                #######################
#pip install gensim
from gensim.corpora.dictionary import Dictionary

#as gensim dictionary takes an array of tokens(list.O.lists) and not single string 
#lets sent tokenize first
from nltk.tokenize import sent_tokenize
sent_tokens=[w for w in sent_tokenize(txt1)]
no_stops=[sent.lower() for sent in sent_tokens if sent not in stopwords.words('english') ]
# this stopwprds removal doesn't work , need to find a way to apply to list.O.lists


word_tokens=[word_tokenize(w) for w in no_stops]

#solution:

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

no_stop = []
for item in word_tokens:
    temp = []
    for word in item:
        if word not in stop_words:
            temp.append(word)
    no_stop.append(temp)

dictionary=Dictionary(no_stop)

#bag of words from dictionary

corpus=[dictionary.doc2bow(doc) for doc in no_stop]
sort_corpus=sorted(corpus,key= lambda w: w[1], reverse = True)

import itertools

from collections import defaultdict
total_word_count=defaultdict(int)
for word_id,word_count in itertools.chain.from_iterable(sort_corpus):
    total_word_count[word_id]+=word_count

sorted_word_count=sorted(total_word_count.items(), key= lambda w:w[1], reverse=True)

for word_id, word_count in sorted_word_count[:15]:
    print(dictionary.get(word_id),word_count)
    
    
    ####################
    ####TFIDF GENSIM####
    ####################

from gensim.models.tfidfmodel import TfidfModel

tfidf=TfidfModel(corpus)

#from just one corpus:

tfidf_weights= tfidf[corpus[0]]
sorted_tfidf=sorted(tfidf_weights,key= lambda w:w[1],reverse=True)
for term_id,weight in sorted_tfidf[:10]:
    print(dictionary.get(term_id),weight)

# to print top 10 1ords from all corpus:
for i in range(0,len(corpus)+1):
    tfidf_weights= tfidf[corpus[i]]
    sorted_tfidf=sorted(tfidf_weights,key= lambda w:w[1],reverse=True)
    for term_id,weight in sorted_tfidf[:10]:
        print(dictionary.get(term_id),weight)


    ################################
    ### Named Entity Recognition ###
    ################################
    


a=nltk.sent_tokenize(txt1)
b=[nltk.word_tokenize(sent) for sent in a]

#applying stop words removal from sublists of a list
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

no_stop = []
for item in b:
    temp = []
    for word in item:
        if word not in stop_words:
            temp.append(word)
    no_stop.append(temp)

c=[nltk.pos_tag(sent) for sent in no_stop]

   
d=nltk.ne_chunk_sents(c)
            
ner_categories=defaultdict(int)
for sent in d:
    for chunk in sent:
        if hasattr(chunk,'label'):
            ner_categories[chunk.label()]+=1
            
labels = list(ner_categories.keys())
values = [ner_categories.get(v) for v in labels]

plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
plt.show()
