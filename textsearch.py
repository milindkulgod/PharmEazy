import numpy as np
import pandas as pd
import pickle
import string 
import html
import ast
import re
import nltk
import time
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
stpwrds = stopwords.words("english")
from nltk.stem import WordNetLemmatizer

class backend():
    def __init__(self):
        self.dataset = pd.read_csv("full1.csv")
        try:
            with open("wordbankdoc.pickle", "rb") as pic:
                self.wordbank = pickle.load(pic)
        except: 
            start = time.time()
            print("\nCreating word bank for the full data\n")
            self.wordbank = scr.InvInd(self.dataset)
            with open('wordbankdoc.pickle',"wb") as p:
                pickle.dump(self.wordbank, p)

    def encode_the_reviews(self,review):
        return html.unescape(review)
	
    def elim_stopword(self,r):
        r_n = " ".join([i for i in r if i not in stpwrds])
        return r_n
        
    def lem(self,tokens):
        l = WordNetLemmatizer()
        out = [l.lemmatize(word) for word in tokens]
        return out

    def InvInd(self,dataset):
        dataset['review']=dataset['review'].apply(str)
        dataset['review']=dataset['review'].apply(self.encode_the_reviews)
        response = dataset['review'].str.replace("[^a-zA-Z]", " ")
        response = response.apply(lambda r: " ".join([w for w in r.split() if len(w)>2]))
        response = [self.elim_stopword(r.split()) for r in response]
        response = [r.lower() for r in response]
        response = pd.Series(response)
        word_tokens = response.apply(lambda r: r.split())
        response = word_tokens.apply(self.lem)
                
        wordbank = {}
                
        for i,r in enumerate(response, start=0):
                for j,w in enumerate(r , start=0):
                        if w not in wordbank:
                            wordbank[w] = [1,{i:[j]}] #add to dictionary when word is new. Stores document id and position at which the word occurs.
                        else:
                            if i not in wordbank[w][1]: #if the word already exists, check for i in the second dictionary, add and increment new document.
                                    wordbank[w][0] += 1
                                    wordbank[w][1][i] = [j]
                            else:
                                    if j not in wordbank[w][1][i]: #if word and document both are there, append to the list.
                                        wordbank[w][1][i].append(j)

        N = np.float64(dataset.shape[0])                    

        for w in wordbank.keys():
            plist = {}
            for i in wordbank[w][1].keys(): #Creating an empty plist dictionary while going through every word. for every word for every document it occurs in.
                tf = (len(wordbank[w][1][i])/len(response[i])) #number of times a term occurs in a document / total number of terms
                weight_i = (1 + np.log10(tf)) * np.log10(N/wordbank[w][0])#idf calculation total docs/no. of documents the word occurs in. Store as list.
                plist[i] = weight_i #every document has a weight. Hence append. List format for appending.
            wordbank[w].append(plist)
        p = open('wordbankdoc.pickle',"wb")
        pickle.dump(wordbank,p) #save the file by dumping


    def topk(self,query):

        q = query.replace("[^a-zA-Z]", " ").lower()
        q_vec = self.elim_stopword(q.split()) #Preprocess input query
        q_vect = self.lem(q_vec.split())
        
        srtdplist = {} #Empty
        qw = {}
        for w in q_vect: #Going through all the words in the query
            if w in self.wordbank.keys(): #if word is there, and if the srtdplist key doesnt have the word
                if w not in srtdplist.keys():
                    srtdplist[w] = sorted(self.wordbank[w][2].items(), key=lambda x:x[1], reverse=True)[:10] #retrieve sorted version of plist. w[2] because appended. Returns a list of tuples.
            if w not in qw: # query weight - weight of word in query. no of times word occurs / length of query
                qw[w] = [1,(1/len(q_vect))] #see word for the first time, initialize count to 1, and weight to eqn.
            elif w in qw: #if word exists, you update
                qw[w][0] += 1 #increment the count. Word occurs for the second time in the query vector.
                qw[w][1] = (qw[w][0]/len(q_vect)) #get the new weight with new count.
        if srtdplist == {}: #compares to vocabulary. If no matches found.
            return "No results found"
        
        topk = [] #empty list to store document id,and weight tuple
        N = self.dataset.shape[0] #total data size 
        for i in range(N): #goes through all document ids to get the count of the document id occurance in the plist
            count = 0 #count of current document id
            sd = 0
            for w in q_vect: 
                for (di,wt) in srtdplist[w]: #for every word, we check document (id, weight) tuple.
                    if di == i: count += 1 #if document id in the list is the current document id, count is incremented
            if count > 0 and count == len(q_vect): #count of document within plists, no. of times it has occured is > 0 for documents that have occured. Has occured in plist. 
                for w in q_vect:
                    l = [x for x in srtdplist[w] if x[0] == i] #current word and document is the weight
                    sd += l[0][1] * qw[w][1] #weight calculation from plist of the word. The final score for the document. Weight * weight of query word is the document score.
                topk.append((i,sd)) #check document id, retrieve the weight.
            elif count > 0 and count < len(q_vec): 
                
                for w in q_vect:
                    l = srtdplist[w][9] #document hasnt occured, partially correct result, give it lesser weight.
                    sd += l[1] * qw[w][1] #giving own weight will give it a significant value
                topk.append((i,sd))  #append to final score
                
        
        show = [x for x in sorted(topk, key=lambda i:i[1], reverse=True)]        
        out = []
        for (ind,s) in show:
             out.append( [self.dataset.loc[self.dataset.index[ind], 'drugName'], self.dataset.loc[self.dataset.index[ind], 'usefulCount'], self.dataset.loc[self.dataset.index[ind], 'condition'], self.dataset.loc[self.dataset.index[ind], 'rating'], self.dataset.loc[self.dataset.index[ind], 'review'], s])
        
        
        pd.set_option('display.max_columns', -1)  
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('max_colwidth', -1)
        out =  pd.DataFrame(out, columns=['Drug Name','Useful count','Condition','Rating(/10)','Review','Similarity%'])
      
        return out  

