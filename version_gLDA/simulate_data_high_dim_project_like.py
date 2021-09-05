
"""
Created on Sat Jun  5 18:55:37 2021

@author: chloepeng
"""

import numpy as np
import scipy.stats as sp
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def simul_data_func(num_topic, num_embed, num_doc, path, data_file):

    alpha = np.array(np.random.uniform(low = 6, high = 10, size = num_topic))
    v = np.load(path+data_file)
    
    kappa =5
    
    user_size_perc = np.array([0.75, 0.15, 0.1])
    num_each_doc_small =  np.array(np.random.uniform(low = 1, high = 6, size = int(user_size_perc[0]*num_doc))).astype(int)
    num_each_doc_med =  np.array(np.random.uniform(low = 6, high = 15, size = int(user_size_perc[1]*num_doc))).astype(int)
    num_each_doc_deep =  np.array(np.random.uniform(low = 16, high = 1000, size = int(user_size_perc[2]*num_doc))).astype(int)
    num_each_doc = np.append(num_each_doc_small, num_each_doc_med)
    num_each_doc = np.append(num_each_doc, num_each_doc_deep)
    
    total_words = np.sum(num_each_doc)
    nu = total_words
    docs = np.repeat(np.array([doc for doc in range(num_doc)]),num_each_doc).reshape((total_words,1))
    doc_map = np.array([doc for doc in range(total_words)]).reshape((total_words,1))
    doc_map = np.concatenate((docs,doc_map),axis=1)
    
    #initialization
    theta = sp.dirichlet.rvs(alpha, size=num_doc)
    theta_words = np.repeat(theta, num_each_doc, axis=0)
    
    z = np.zeros((total_words, 1)).astype(int)
    for word in range(total_words):
        z[word,0] = np.argmax(sp.multinomial.rvs(1,theta_words[word,:]))
    
    kmeans = KMeans(n_clusters=num_topic, random_state=0).fit(v)
    labels = kmeans.labels_
    
    sigmas = np.zeros((num_topic, num_embed, num_embed))
    mus = np.zeros((num_topic, num_embed))
    for topic in range(num_topic):
        mus[topic, :]=np.mean(v[labels==topic, :], axis =0)
        sigmas[topic, :, :] = np.cov(v[labels==topic, :].T)*(v[labels==topic, :].shape[0]/total_words)
    mu_0 = np.mean(mus, axis =0)
    sigma_mean = np.mean(sigmas)+np.identity(num_embed)
    
    psi = np.linalg.inv(sigma_mean*(1/(nu+2)))
    
    sigma = sp.invwishart.rvs(df=nu, scale=psi, size=num_topic)
    eigens = np.zeros((num_topic, num_embed, num_embed))
    for topic in range(num_topic):
        _, eigens[topic, :, :], _ = np.linalg.svd(sigma[topic,:,:])
    
    
    mu = np.zeros((num_topic, num_embed))
    for topic in range(num_topic):
        mu[topic,:] = np.random.multivariate_normal(mu_0, (1/kappa)*sigma[topic,:,:], size = 1)
        
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
             'mu':mu,
             'sigma':sigma,
             'nu':nu,
             'kappa':kappa,
             'psi':psi,
    
             'alpha':alpha,
             'theta':theta,
             'theta_words':theta_words,
             'z':z,
             'v':v}
    
    np.save(path+'simul_data/simul_'+str(num_embed)+'_k_'+str(num_topic) + '_doc_'+str(num_doc)+'_kappa_'+str(kappa), model)
    
if __name__ == 'main':
    path = "../../data/"
    num_topic =10
    num_embed =585
    num_doc = 1300
    data_file ='processed_real_data/vocab_embed_47870.npy'
    simul_data_func(num_topic, num_embed, num_doc, path, data_file)