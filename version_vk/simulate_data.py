
"""
Created on Sat Jun  5 18:55:37 2021

@author: chloepeng
"""

import numpy as np
import scipy.stats as sp

path = "../../data/simul_data/"


#prior 
num_topic = 2
num_embed = 3
num_doc = 3

alpha = np.ones(num_topic)
mu_0 = np.array([3, 1, 2])

nu_V = num_embed+1
psi_V = np.identity(num_embed)

nu_sigma = num_embed+4
psi_sigma = np.identity(num_embed)

num_each_doc = np.array([100, 200, 50])
total_words = 0
for doc in range(num_doc):
    total_words = total_words + num_each_doc[doc]
docs = np.repeat(np.array([doc for doc in range(num_doc)]),num_each_doc).reshape((total_words,1))
doc_map = np.array([doc for doc in range(total_words)]).reshape((total_words,1))
doc_map = np.concatenate((docs,doc_map),axis=1)

#initialization
theta = sp.dirichlet.rvs(alpha, size=num_doc)
theta_words = np.repeat(theta, num_each_doc, axis=0)

z = np.zeros((total_words, 1)).astype(int)
for word in range(total_words):
    z[word,0] = np.argmax(sp.multinomial.rvs(1,theta_words[word,:]))

V = sp.invwishart.rvs(df=nu_V, scale=psi_V, size=num_topic)
mu = np.zeros((num_topic, num_embed))
for topic in range(num_topic):
    mu[topic,:] = np.random.multivariate_normal(mu_0, V[topic,:,:], size = 1)

sigma = sp.invwishart.rvs(df=nu_sigma, scale=psi_sigma, size=num_topic)
v = np.zeros((total_words,num_embed))
for word in range(total_words):
    v[word,:] = np.random.multivariate_normal(mean = mu[z[word,0],:], cov = sigma[z[word,0],:,:], size = 1)

for topic in range(num_topic):
    print('\nk ==',topic)
    print('mu')
    print(np.mean(v[z[:,0]==topic,:],axis=0))
    print(mu[topic,:])
    print('\nsigma')
    print(np.cov(v[z[:,0]==topic,:].T))
    print(sigma[topic,:,:])

# t-test for differences if k == 2
for embed in range(num_embed):
    print('embedded dim:',embed)
    mean0 = np.mean(v[z[:,0]==0,embed],axis=0)
    mean1 = np.mean(v[z[:,0]==1,embed],axis=0)
    print('means:',mean0,mean1,mean1-mean0)
    std0 = np.cov(v[z[:,0]==0,embed].T)**0.5
    std1 = np.cov(v[z[:,0]==1,embed].T)**0.5
    print('stds:',std0,std1,(std0+std1)/2)
    print('diffs_in_std:',(mean1-mean0)/((std0+std1)/2))

model = {'num_topic':num_topic,
         'num_embed':num_embed,
         'num_doc':num_doc,
         'num_each_doc':num_each_doc,
         'total_words':total_words,
         'doc_map':doc_map,
         
         'mu_0':mu_0,
         'nu_V':nu_V,
         'psi_V':psi_V,
         'nu_sigma':nu_sigma,
         'psi_sigma':psi_sigma,
         'mu':mu,
         'V':V,
         'sigma':sigma,
         
         'alpha':alpha,
         'theta':theta,
         'theta_words':theta_words,
         'z':z,
         'v':v}

np.save(path+'simul_model', model)