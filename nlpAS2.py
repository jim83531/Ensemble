# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 14:52:41 2018

@author: Sid007
"""

import xml.etree.cElementTree as et
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


def getvalueofnode(node):
    """ return node text or None """
    return node.text if node is not None else None

def main():
    """ main """
    parsed_xml = et.parse("Restaurants_Train.xml")
    root = parsed_xml.getroot()
    dfcols0 = ['text']
    dfcols1 = ['term']
    dfcols2 = ['polarity']
    text_t = pd.DataFrame(columns=dfcols0)
    term_t = pd.DataFrame(columns=dfcols1)
    polarity_t = pd.DataFrame(columns=dfcols2)
    
    for node in root:
        text = node.find('text')
        text_t = text_t.append(pd.Series([getvalueofnode(text)], index=dfcols0),ignore_index=True)
        
    for elem in root.iterfind('sentence/aspectTerms/aspectTerm'):
        term = elem.get('term')
        polarity = elem.get('polarity')
        term_t = term_t.append(pd.Series([term], index=dfcols1),ignore_index=True)   
        polarity_t = polarity_t.append(pd.Series([polarity], index=dfcols2),ignore_index=True)
            
        
        
    
    text_t.to_csv("text_t.csv", sep=',',index = False)
    term_t.to_csv("term_t.csv", sep=',',index = False)
    polarity_t.to_csv("polarity_t.csv", sep=',',index = False)
    y_term = term_t.values
    y_polarity = polarity_t.values
    
    n,d = y_term.shape
    print(y_term.shape)
    y_term = y_term.reshape(n,1)

    m,k = y_polarity.shape
    print(y_polarity.shape)
    y_polarity = y_polarity.reshape(m,1)
    
    print(text_t.shape)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(text_t['text'])
    print(X_train_counts.shape)
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    print(X_train_tf.shape)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    print(X_train_tfidf.shape)
    
    #clf_term = MultinomialNB().fit(X_train_tfidf, y_term)
    #clf_polarity = MultinomialNB().fit(X_train_tfidf, y_polarity)
    
 
main()