#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import random
import string


# In[2]:


f=open('Desktop/computers.txt','r', errors='ignore')
data=f.read()
data=data.lower()
nltk.download('punkt')
nltk.download('wordnet')
sent=nltk.sent_tokenize(data)
word=nltk.word_tokenize(data)


# In[3]:


greetings=['Hello','Hi','Hey','Yo','How are you','Good morning','Good evening','Good afternoon','Hi there']
responses=['Hi','Hello','Hey','I am good','Hi.How are you']

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greetings:
            return random.choice(responses)

print(greet('I am not feeling well, but Hello') )       
        


# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[5]:


def response(user_response):
    bot_response=''
    sent.append(user_response)
    TfidfVec = TfidfVectorizer( stop_words='english')
    tfidf = TfidfVec.fit_transform(sent)

    print(tfidf[-1])
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
            bot_response=bot_response+"I am sorry! I don't understand you"
            return bot_response
    else:
            bot_response = bot_response+sent[idx]
            return bot_response


# In[ ]:





# In[6]:


flag=True
print("Hi. Please type in your questions about computers. Type 'bye' to exit.")
while(flag==True):
    user_response=input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thank you' or user_response=='thanks'):
            flag=False
            print("You're welcome!")
        else:
            if (greet(user_response)!=None):
                print(greet(user_response))
            else:
                print(response(user_response))
                sent.remove(user_response)
    else:
        flag=False
        print("Bye!See you soon!")
        


# In[ ]:




