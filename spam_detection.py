#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 07:41:53 2023

@author: fibonacci
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
import re
dataset=pd.read_csv('spam_ham_dataset.csv')
dataset=dataset.drop(columns=['label','Unnamed: 0'],axis=1)
print(string.punctuation)

def remove_punctuatio(text):
    text_notpun="".join([i for i in text if i not in string.punctuation])
    return text_notpun

dataset['clean_text']=dataset['text'].apply(lambda x:remove_punctuatio(x))


###tokenization
def tokenization(text):
    token=re.split('\W+', text)
    return token

dataset['clean_text']=dataset['text'].apply(lambda x: tokenization(x.lower()))

stopwords=nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    text_notpun="".join([i for i in text if i not in stopwords])
    return text_notpun

dataset['clean_textsr']=dataset['clean_text'].apply(lambda x:remove_stopwords(x))
