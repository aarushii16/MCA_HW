#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sklearn
import math
import nltk
import re
from collections import defaultdict
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE


# In[2]:


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('abc')
nltk.download('punkt')


# In[3]:


# SOFTMAX ACTIVATION FUNCTION
def softmax(x):
    m=max(x)
    xx=x-m
    e=np.exp(xx)
    s=np.sum(e,axis=0)
    ans=e/s
    return ans


# In[4]:


def norm(v):
    sq=0
    for i in v:
        sq=sq+i*i
    return math.sqrt(sq)


# In[5]:


def outer_prod(x,y):
    ans=[[0 for i in range(len(y))] for j in range(len(x))]
    for i in range(len(x)):
        for j in range(len(y)):
            ans[i][j]=x[i]*y[j]
    return np.asarray(ans)


# In[6]:


def uniform_random(a,b):
    r=random.uniform(a,b)
    while r==b:
        r=random.uniform(a,b)
    return r


# In[7]:


def rand_matrix(l,u,r,c):
    ans=[[0 for i in range(c)] for j in range(r)]
    for i in range(r):
        for j in range(c):
            ans[i][j]=uniform_random(l,u)
    return np.asarray(ans)


# In[8]:


def lse(u):
    a=np.exp(u)
    b=a.sum()
    c=np.log(b)
    return c


# In[53]:


class word2vec():
    def __init__ (self,n,lr,e,w):
        self.n = n
        self.eta = lr
        self.epochs = e
        self.window = w
        self.training_data=[]
        pass
    
    def generate_training_data(self, corpus):

        word_counts = {}
        for row in corpus:
            for word in row:
                try:
                    word_counts[word] = word_counts[word] + 1
                except:
                    word_counts[word] = 1

        self.v_count = len(word_counts)
#         print(self.v_count)

        tem=list(word_counts.keys())
        self.words_list = sorted(tem)
        self.word_index={}
        self.index_word={}
#         print(self.words_list)
        
        for i in range(len(self.words_list)):
            self.word_index[self.words_list[i]]=i
            self.index_word[i]=self.words_list[i]

        training_data = []
        for sentence in corpus:
            ls = len(sentence)

            for i in range(ls):
                
                target = self.word2onehot(sentence[i])

                context = []
                winzie=self.window
                for j in range(i-winzie, i+1+winzie):
                    tem=0
                    if 0<=j and j<=ls-1:
                        if j!=i:
                            tem=self.word2onehot(sentence[j])
                            context.append(tem)
#                         tem=1
                
                training_data.append([target, context])
        
        return np.array(training_data) 
    

    def word2onehot(self, word):
        word_vec=[]
        word_index = self.word_index[word]
        for i in range(self.v_count):
            if i!=word_index:
                word_vec.append(0)
            else:
                word_vec.append(1)
        return word_vec


    def forward_pass(self, x):
        w11=self.w1.T
        w22=self.w2.T
        h = np.dot(w11, x)
        u = np.dot(w22, h)
        y_c = softmax(u)
        return y_c, h, u
                

    def update_weights(self,d1,d2):
        self.w1 = self.w1 - (self.eta * d1)
        self.w2 = self.w2 - (self.eta * d2)
    
    def backprop(self, e, h, x):
        dl_dw2 = outer_prod(h, e)
        tem=np.dot(self.w2, e.T)
        dl_dw1 = outer_prod(x, tem)
        self.update_weights(dl_dw1,dl_dw2)


    def train(self, training_data_whole):
#         if extend==False:
        self.w1 = rand_matrix(-0.5, 0.5, self.v_count, self.n)     # embeddings
        self.w2 = rand_matrix(-0.5, 0.5, self.n, self.v_count)     # context
        
#         for zz in range(100):
#             try:
#                 training_data=training_data_whole[int(zz*100):int(100*(zz+1))]
#             except:
#                 training_data=training_data_whole[int(zz*100):]
        
    training_data=training_data_whole
    for i in range(self.epochs):

        self.loss = 0

        for w_t, w_c in training_data:

            y_pred, h, u = self.forward_pass(w_t)

            xy=[]
            for t in range(len(w_c)):
                w=w_c[t]
                tt=(y_pred-w).astype('float')
                xy.append(tt)
            xy=np.asarray(xy)

            EI = xy.sum(axis=0)

            self.backprop(EI, h, w_t)

            uu=lse(u)
            ab=[]
            for t in range(len(w_c)):
                ind=w_c[t].index(1)
                temp=u[ind]
                ab.append(temp)
            ab=np.asarray(ab)
            self.loss += -np.sum(ab) + len(w_c) * uu

        print ('EPOCH:',i, 'LOSS:', self.loss)
        if i%10==0:
            visualize(i)
        pass

    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def word_sim(self, word, top_n):
        vc=self.v_count
        ww=self.word_index
        vw=self.w1
        w1_index = ww[word]
        v_w1 = vw[w1_index]

        word_sim = {}
        for i in range(vc):
            tt=self.w1
            v_w2 = tt[i]
            theta= np.dot(v_w1, v_w2)/(norm(v_w1) * norm(v_w2))
            iw=self.index_word
            word = iw[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv:(kv[1], kv[0]),reverse=True)
        return words_sorted[:top_n]


# In[54]:


#code taken from the reference link given
def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=color, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=8)
#     plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    if filename:
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# In[55]:


#code taken from the reference link given
def visualize(i):
    embedding_clusters = []
    word_clusters = []
    for row in corpus:
        for word in row:
            embeddings = []
            words = []
            for t in w2v.word_sim(word, 5):
                similar_word=t[0]
                words.append(similar_word)
                embeddings.append(w2v.word_vec(similar_word))
            embedding_clusters.append(embeddings)
            word_clusters.append(words)
    embedding_clusters = np.array(embedding_clusters)
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
    embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))).reshape(n, m, 2)
    name='similar_words'+str(i)+'.png'
    tsne_plot_similar_words('Similar words', corpus, embeddings_en_2d, word_clusters, 0.7,name)



corpus = nltk.corpus.abc.sents()
w2v = word2vec(5,0.01,1000,2)
training_data = w2v.generate_training_data(settings, corpus)
w2v.train(training_data)
