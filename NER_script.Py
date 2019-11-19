# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

f = open('C:\\Users\\sachin.kalra\\Desktop\\news_article.docx', 'rb')

f=open('C:\\Users\\sachin.kalra\\Desktop\\news_art.txt','rb')

news_content=f.read()


nltk.download()
import nltk


results=[]


nltk.sent_tokenize(news_content)

import nltk
for sent_no,sentence in enumerate(nltk.sent_tokenize(news_content)):
    no_of_tokens=len(nltk.word_tokenize(sentence))
    tagged=nltk.pos_tag(nltk.word_tokenize(sentence))
    no_of_nouns=len([word for word,pos in tagged if pos in ["NN","NNP"]])
    ners=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)),binary=False)
    #ps=nltk.pos_tag(nltk.word_tokenize(sentence))
    #print(ners[1])
    #print('------------')
    #print(ps[1])
    no_of_ners=len([chunk for chunk in ners if hasattr(chunk,'label()')])
    score=(no_of_ners+no_of_nouns)/float(no_of_tokens)
    results.append((sent_no,no_of_tokens,no_of_ners,no_of_nouns,score,sentence))


nltk.set_proxy('http://sachin.kalra:India@06@http://cpinternet.com:8080')
nltk.download()



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
a=nltk.sent_tokenize(txt)
b=[nltk.word_tokenize(sent) for sent in a]
c=[nltk.pos_tag(sent) for sent in b]

d=nltk.ne_chunk_sents(c,binary=True)

for sent in d:
    #print("the sent is: ",d)
    for chunk in sent:
        #print("the chunk in above sent is:  ",chunk)
        if hasattr(chunk, "label"): 
            #print(type(chunk))
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

