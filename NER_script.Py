# -*- coding: utf-8 -*-
"""
Spyder Editor
Script for Named Entity Recognition 

"""
txt=(''' about 100 anti-government protesters remained holed up at a Hong Kong
     university Tuesday as a police siege of the campus entered its third 
     day.City leader Carrie Lam said 600 people had left the Hong Kong 
     Polytechnic campus, including 200 who are under 18 years old.Police 
     have surrounded the university and are arresting anyone who leaves. 
     Lam said those under 18 would not be immediately arrested but could 
     face charges later.She said that the other 400 who have left have been 
     arrested."We will use whatever means to continue to persuade and 
     arrange for these remaining protesters to leave the campus as soon 
     as possible so that this whole operation could end in a peaceful manner,
     " she said after a weekly meeting with advisers.Now in its fifth 
     month, the Hong Kong protest movement has steadily intensified as 
     local and Beijing authorities harden their positions and refuse to
     make concessions.Universities have become the latest battleground for
     the protesters, who used gasoline bombs and bows and arrows in their 
     fight to keep riot police backed by armoured cars and water cannon out 
     of two campuses in the past week.China, which took control of the 
     former British colony in 1997 promising to let it retain its autonomy,
     flexed its muscles, sending troops outside their barracks over the 
     weekend in a cleanup operation.China's ambassador to Britain accused 
     the UK and the US of meddling in the country's internal affairs and
     warned that the Chinese government "will not sit on our hands" if the 
     situation in Hong Kong "becomes uncontrollable "Lam, asked whether 
     she would seek help from Chinese troops based in Hong Kong, said her 
     government remains confident it is able to cope with the situation''')

import nltk
from collections import defaultdict
import matplotlib.pyplot as plt

a=nltk.sent_tokenize(txt1)
b=[nltk.word_tokenize(sent) for sent in a]
c=[nltk.pos_tag(sent) for sent in b]

d=nltk.ne_chunk_sents(c,binary=True)
for sent in d:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE" :
            print(chunk)
            
# to print only NE , flag Bianry = True and remove chunk.label()
d=nltk.ne_chunk_sents(c)
for sent in d:
    for chunk in sent:
        if hasattr(chunk, "label") :
            print(chunk)            


# relvant 2:
# chunked without binary category names
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


# SpaCy Implementation

import spacy

# this didn't work
#nlp=spacy.load('en')
nlp=spacy.load('en_core_web_sm')

doc=nlp(txt1)

doc.ents
#summarizing o/p 1
x,y=zip(*[(ent.label_,ent.text) for ent in doc.ents])

#summarizing o/p 2
labels=[]
text=[]
for ent in doc.ents:
    labels.append(ent.label_)
    text.append(ent.text)
    
#summarizing o/p 3
import pandas as pd
pd.concat([pd.DataFrame(labels),pd.DataFrame(text)],axis=1)
    
