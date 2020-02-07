# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 19:10:04 2020

@author: sachin.kalra
"""

text="""US forces  have killed the head of Iran's elite Quds Force and the deputy leader of Iraq's Shiite-dominated Popular Mobilization Units.
Days earlier US President Donald Trump warned that Iran would "pay a very big price" after Shiite militia supporters stormed the US Embassy in Baghdad.
Iranian officials have vowed retribution for Friday's attack.
The United States has urged its citizens to leave Iraq "immediately," and the Netherlands has said the same for Dutch nationals. France urged its citizens in neighboring Iran to stay away from public gatherings while Germany advised its nationals not to make video or audio recordings of demonstrations as it may be interpreted as espionage.
Both Israel's Prime Minister Benjamin Netanyahu and US Secretary of State Mike Pompeo have said Qassem Soleimani was planning to carry out more attacks on innocent victims.
Some 3,000 US troops are being sent to the Middle East as a "precautionary" measure.
Read more: US kills top Iranian General Soleimani in Baghdad airstrike

All updates in Coordinated Universal Time (UTC)

03:09  US Secretary of State Mike Pompeo expressed frustration at the perceived lack of support from Washington's European allies. "Frankly, the Europeans haven't been as helpful as I wish that they could be. The Brits, the French, the Germans all need to understand that what we did, what the Americans did, saved lives in Europe as well," Pompeo said in an interview with Fox News.

"This was a good thing for the entire world, and we are urging everyone in the world to get behind what the United States is trying to do to get the Islamic Republic of Iran to simply behave like a normal nation," he added.

01:19 German Greens politician Omid Nouripour has warned that "the risk of an escalation is very high," with the danger Iran's armed forces could "hit the Americans and us, as their allies, massively."

In an interview with DW, Nouripour said it was time to reactivate "any imaginable mechanism of crisis diplomacy," including sending Germany's foreign minister to Tehran, Riyadh, Doha and Ankara to find a way "to prevent an overreaction to a potential Iranian overreaction." 

He added that German soldiers should be taken out of Iraq immediately because security on the ground cannot be guaranteed."""


import nltk

sents=nltk.sent_tokenize(text)

words=[nltk.word_tokenize(i) for i in sents]


#stemming

from nltk import PorterStemmer

stemmer=PorterStemmer()

#1 : Stemming
sents=nltk.sent_tokenize(text)

for i in range(len(sents)):
    words=nltk.word_tokenize(sents[i])
    newwords=[stemmer.stem(word) for word in words]
    sents[i]=' '.join(newwords)

    
#2 : Lemmatization 
    
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

sents=nltk.sent_tokenize(text)

for i in range(len(sents)):
    words=nltk.word_tokenize(sents[i])
    newwords=[lemmatizer.lemmatize(word, pos = 'v') for word in words]
    sents[i]=' '.join(newwords)

# note: need to mention POS in lemmatize() about which POS lemmatization is required.
    
    
#3 : Stop Word Removal

nltk.download('stopwords')
from nltk.corpus import stopwords

sent=nltk.sent_tokenize(text)

for i in range(len(sent)):
    word=nltk.word_tokenize(sent[i])
    newwords=[w for w in word if w not in stopwords.words('english')]
    sent[i]= ' '.join(newwords)
    
    
#4 POS tagging:
    
words=nltk.word_tokenize(text)

tagged_words=nltk.pos_tag(words)

word_tags=[]

for tw in tagged_words:
    word_tags.append(tw[0]+"_"+tw[1])
    
tagged_paragraph= ' '.join(word_tags)

#5 POS Tagging Datacamp 

sent=nltk.sent_tokenize(text)
words=[nltk.word_tokenize(i) for i in sent]

tw=[nltk.pos_tag(w) for w in words]
tw_tree=nltk.ne_chunk_sents(tw,binary=True)


for tw in tw_tree:
    for chunk in tw:
        if hasattr(chunk,"label") and chunk.label()=='NE':
            print(chunk)

#5 NER Tagging
            
words=nltk.word_tokenize(text)
tagged_words=nltk.pos_tag(words)

namedEnt=nltk.ne_chunk(tagged_words)
namedEnt.draw()


#6 Bag of Words Model-Most common words datacamp 

from nltk import Counter
from nltk.corpus import stopwords
# so far we already have text, lets do a quick bow model:


alpha_tokens_lower= [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
no_stops=[w for w in alpha_tokens_lower if w not in stopwords.words('english') ]


len(alpha_tokens_lower)-len(no_stops)
words_count=Counter(no_stops)

words_count.most_common(17)

#7 Bag of words model Udemy:

#we already have the text:
import re
dataset=nltk.sent_tokenize(text)


#lowercase, non-alpha to singe space
for i in range(len(dataset)):
    dataset[i]=dataset[i].lower()
    dataset[i]=re.sub(r'\W', ' ', dataset[i])
    dataset[i]=re.sub(r'\s+',' ',dataset[i])  
    



