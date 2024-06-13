#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Student id: 334371478


# # (i) Data Preparation

# In[1]:


import os
import nltk
import re
import numpy as np


# In[2]:


def read_file(path):
    with open(path) as f: 
        return f.readlines()
dirs = [dr for dr in os.scandir(r"D:\bbc") if 
        dr.is_dir()]
# Store texts categorized by directory
texts = {}
for dr in dirs:
    category = dr.name
    files = [file.path for file in os.scandir(dr) if file.is_file()]
    text = [read_file(file) for file in files]
    texts [category] = text


# In[3]:


category = list(texts.keys())
print(category) # categories in texts


# In[4]:


sum(len(texts[category]) for category in texts) #documents in texts


# In[5]:


nltk.download("punkt")


# In[6]:


nltk.download("averaged_perceptron_tagger")
nltk.download("tagsets")


# In[7]:


#lemmatizer and stopwords
from nltk.corpus import stopwords
stops = stopwords.words("english")

from nltk.stem import WordNetLemmatizer
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()


# In[8]:


from nltk import sent_tokenize, word_tokenize
def extract_words(text):
    words = [word.lower() # lowercase
             for word in re.findall("[^\d\W]+", text)] # no punctuation
    return words
def filter_words(words, stops, lemmatizer):
    filtered_words = [word for word in words 
                      if word not in stops] # no stopwords
    lemmatized_words = [lemmatizer.lemmatize(word) 
                        for word in filtered_words] # lemmatized words
    return lemmatized_words
def tokenise(text, stops, lemmatizer):
    return filter_words(extract_words(text), stops, lemmatizer) # do two prev funct


# In[9]:


from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
tokeniser_partial = partial(tokenise, lemmatizer=lemmatizer, stops=stops)
vectorizer = CountVectorizer(analyzer=tokeniser_partial)


# In[10]:


# process for each documents
def stream_corpus(corpus):
    for category in corpus.values():
        for article in category:
            yield ' '.join(article)
texts_streamer = stream_corpus(texts)


# In[11]:


texts_streamer


# In[12]:


vectorizer.fit(texts_streamer)


# In[13]:


terms = vectorizer.get_feature_names_out()


# In[14]:


DD = vectorizer.transform(stream_corpus(texts))
DD


# In[15]:


from sklearn.feature_extraction.text import TfidfTransformer
tfidf_array = TfidfTransformer().fit_transform(DD)
tfidf_array


# In[16]:


#--------------------------------------------------------------------------------------------------


# In[ ]:





# # (ii) Lower Dimensional and Similarity

# In[17]:


from sklearn.decomposition import TruncatedSVD
DD = tfidf_array
lsa = TruncatedSVD(n_components=50)
lsa.fit(DD)


# In[18]:


lsa.explained_variance_


# In[19]:


lsa_obj = TruncatedSVD(n_components=50).fit(DD)
D_dash = lsa_obj.transform(DD)
D_dash.shape


# In[20]:


import pandas as pd
lsa_df = pd.DataFrame(D_dash)


# In[21]:


# Add category and article number as index
index_values = []
for category, articles in texts.items():
    for idx, article in enumerate(articles, 1):  # Start index from 1
        index_values.append(f"{category}_{idx}")
lsa_df.index = index_values
display(lsa_df)


# In[22]:


from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between the first politics article to other articles
sim_scores = []
for index, row in lsa_df.iterrows():
    similarity = cosine_similarity([lsa_df.iloc[896]], [row])[0][0] #first politic article
    sim_scores.append((index, similarity))
sim_scores.sort(key=lambda x: x[1], reverse=True) # Sort the similarity scores
# Select the top five articles with the highest similarity scores
top_five = sim_scores[1:6] #exclude the reference
# Print the result
for i, (index, similarity) in enumerate(top_five, 1):
    print(f"article: {index}")
    print(f"Similarity Score: {similarity}")
    print()


# In[23]:


#--------------------------------------------------------------------------------------------------


# In[ ]:





# # (iii) Extracting topics and important terms

# In[24]:


from sklearn.decomposition import NMF
nmf = NMF(n_components = 6,random_state=123)
nmf.fit(DD)


# In[25]:


nmf.components_.shape


# In[26]:


nmf_term_df = pd.DataFrame(nmf.components_.T)
nmf_term_df["terms"] = terms


# In[27]:


nmf_term_df


# In[28]:


for i, col in enumerate(nmf_term_df.columns[:-1]):
    important_terms = " , ".join(nmf_term_df.sort_values(col, ascending=False).terms.head(10)) # important terms for each 
    topic_num = i + 1
    print(f"Topic {topic_num}")
    print(f"Important terms: {important_terms}")
    print()


# In[29]:


#--------------------------------------------------------------------------------------------------


# In[ ]:





# # (iv) Single topic label for each document

# In[30]:


category = list(texts.keys())
print(category) #categories in texts


# In[31]:


doc_topics = nmf.transform(DD) # Transform the TF-IDF matrix into the topic space
#Looking for the dominant topic
doc_topics_df = pd.DataFrame(doc_topics, columns=[f"Topic_{i+1}" for i in range(6)])
doc_topics_df["dominant_Topic"] = doc_topics.argmax(axis=1) + 1 


# In[32]:


# Add the initial category of the articles
cat = []
for category, articles in texts.items():
    for idx, article in enumerate(articles):  
        cat.append(f"{category}")
doc_topics_df ["category"] = cat


# In[33]:


# add column name to the dominant topics and the coloumn name
doc_topics_df.columns = ["1: sports","2: politics","3: technology","4: film","5: economy","6: russian","dominant_topic","category"]
# mapping the dominat topic
topic_map = {1: "1: sports", 2: "2: politics",3: "3: tecnology",4: "4: film",5: "5: economy",6: "6: russian"}
doc_topics_df["dominant_topic"] = doc_topics_df["dominant_topic"].map(topic_map)


# In[34]:


doc_topics_df


# In[35]:


# cross-tabulation table
topic_category_cross = pd.crosstab(doc_topics_df["dominant_topic"], doc_topics_df["category"])
topic_category_cross


# In[ ]:





# In[ ]:





# In[ ]:




