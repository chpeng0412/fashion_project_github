#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 18:55:37 2021

@author: chloepeng
"""
import numpy as np
import scipy.stats as sp
import time
import matplotlib.pyplot as plt
from numpy.random import choice


class gLDA_nonCollapsed_lowDim_log():
    """
    Non-Collapsed gLDA model for low dimensional data. To handle overflow issue, sample z in log space.
    """
    
    def __init__(self, iters, path, result_path, simul_data):
        #load data
        self.path = path
        self.result_path = result_path
        self.simul_data = simul_data
    
        #prior 
        self.num_topic = self.simul_data['num_topic']
        self.num_embed = self.simul_data['num_embed']
        self.alpha = self.simul_data['alpha']
        self.mu_0 = self.simul_data['mu_0']
        self.nu_V = self.simul_data['nu_V']
        self.psi_V = self.simul_data['psi_V']
        self.nu_sigma = self.simul_data['nu_sigma']
        self.psi_sigma  = self.simul_data['psi_sigma']
    
        #initialization
        self.num_doc = self.simul_data['num_doc']
        self.total_words = self.simul_data['total_words']
        self.doc_map = self.simul_data['doc_map']
        self.iters = iters
        self.v = self.simul_data['v']
        self.num_each_doc= self.simul_data['num_each_doc']
    
        self.thetas = np.zeros((self.iters, self.num_doc, self.num_topic))
        self.zs = np.zeros((self.iters, self.total_words, 1))
        self.sigmas = np.zeros((self.iters, self.num_topic, self.num_embed, self.num_embed))
        self.Vs = np.zeros((self.iters, self.num_topic, self.num_embed, self.num_embed))
        self.mus = np.zeros((self.iters, self.num_topic, self.num_embed))
    
        
        # self.thetas[0,:,:] = self.simul_data['theta']
        
        self.zs[0,:,:] = choice([i for i in range(self.num_topic)], self.total_words, p=[0.8,0.2]).reshape(-1, 1)
        self.sigmas[0,:,:,:] = self.simul_data['sigma']
        self.Vs[0,:,:,:] = self.simul_data['V']
        self.mus[0,:,:] = self.simul_data['mu']
        
        theta = sp.dirichlet.rvs(self.alpha, size=self.num_doc)
        self.thetas[0,:,:] =  theta
        
        theta_words = np.repeat(theta, self.num_each_doc, axis=0)
        z = np.zeros((self.total_words, 1)).astype(int)
        for word in range(self.total_words):
            z[word,0] = np.argmax(sp.multinomial.rvs(1,theta_words[word,:]))
        self.zs[0,:,:] = z
        
        self.sigmas[0,:,:,:] = np.array([[0.1,0.1,0.1], [0.1,0.5,0.1], [0.1,0.1,0.9]])
        self.Vs[0,:,:,:] =np.array([[1,0,0], [0,1,0], [0,0,2]])
        self.mus[0,:,:] = self.mu_0
    
    def sample_theta(self,i):
        for doc in range(self.num_doc):
            doc_index = self.doc_map[:,0] == doc
            z_doc = self.zs[i-1,doc_index,:]
            _, zks = np.unique(z_doc,return_counts=True)
            self.thetas[i,doc,:] = sp.dirichlet.rvs(self.alpha + zks, size=1)

    def sample_z(self,i):
        inv_sigma = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        log_det = np.zeros((self.num_topic, 1))
        for topic in range(self.num_topic):
            sigma_k = self.sigmas[i-1,topic,:,:]
            #using psedo-inverse here 
            inv_sigma[topic, :, :] = np.linalg.inv(sigma_k)        
            L = np.linalg.cholesky(sigma_k +np.identity(self.num_embed) )
            diag = np.diag(L)
            log_det[topic, 0]= 2*np.sum(np.log(diag))   
    
        
        for word in range(self.total_words):
            vdn = self.v[word,:]
            doc_type = self.doc_map[word,0]
            zk_log = np.zeros((1,self.num_topic))
            zk = np.zeros((1, self.num_topic))
            for topic in range(self.num_topic):
                mu_k = self.mus[i-1,topic,:]
                theta_k = self.thetas[i,doc_type,topic]
                quad_term = np.matmul(np.matmul((vdn-mu_k),inv_sigma[topic, :, :]),(vdn-mu_k))
                zk_log[0,topic] = np.log(theta_k) -(0.5*log_det[topic,0])-(0.5*quad_term)
                
            zk_log_max = np.max(zk_log)
            zk_log_diff = zk_log- zk_log_max
            zk_non_log = np.exp(zk_log_diff)
            zk_sum = np.sum(zk_non_log)
            zk= zk_non_log/zk_sum
            if np.sum(zk)>1:
                diff = np.sum(zk)-1
                max_index = np.argwhere(zk[0, :] == np.max(zk)).item()
                zk[0,max_index] =  zk[0,max_index]  -diff
            self.zs[i,word,:] = np.argmax(sp.multinomial.rvs(1,zk[0,:]))
                
            
    def sample_V(self,i):
        for topic in range(self.num_topic):
            nu_new = self.nu_V + 1
            mu_k = self.mus[i-1,topic,:]
            mu_diff = mu_k - self.mu_0
            psi_new = self.psi_V + mu_diff.reshape(self.num_embed,1) @ mu_diff.reshape(1,self.num_embed)
            self.Vs[i,topic,:,:] = sp.invwishart.rvs(df=nu_new, scale=psi_new, size=1)
    
    def nks_func(self, i):
        values, nks = np.unique(self.zs[i, :, :], return_counts = True)
        return nks
        
    def sample_mu(self,i, nks):
  
        for topic in range(self.num_topic):
            vdnk = np.sum(self.v[self.zs[i, :, 0] == topic], axis = 0)
            mu_k_hat = vdnk/nks[topic]
            V_k_inv = np.linalg.inv(self.Vs[i, topic, :, :])
            sigma_k_inv = np.linalg.inv(self.sigmas[i-1, topic, :, :])
            sigma_new = np.linalg.inv(nks[topic]*sigma_k_inv + V_k_inv)
            mu_new = nks[topic]*np.matmul(sigma_new, np.matmul(sigma_k_inv, mu_k_hat)) + np.matmul(sigma_new, np.matmul(V_k_inv, self.mu_0))
            self.mus[i,topic, :] = np.random.multivariate_normal(mu_new, sigma_new, size = 1)
    
    def sample_sigma(self,i, nks):
        for topic in range(self.num_topic):
            vk = self.v[self.zs[i, :, 0] == topic]
            mu_diff = vk - self.mus[i, topic, :]
            quar_prod = mu_diff.T @ mu_diff
            psi_sigma_new = self.psi_sigma + quar_prod
            nu_sigma_new = self.nu_sigma+nks[topic]
            self.sigmas[i, topic, :, :] = sp.invwishart.rvs(df=nu_sigma_new, scale=psi_sigma_new, size=1)
    
    def sampling(self):
        
        for i in range(1,self.iters):
            print(i)
            
            # sample theta
            self.sample_theta(i)
            
            # sample z
            self.sample_z(i)
        
            
            # sample V
            self.sample_V(i)
        
            nks=self.nks_func(i)
            # sample mu
            self.sample_mu(i, nks)
            
            # sample sigma
            self.sample_sigma(i, nks)


if __name__ == "__main__":
    path = "../../data/simul_data/"
    result_path = "../../results/log_low_dim"
    simul_data = np.load(path+'/simul_model.npy', allow_pickle = True).item()
    iters =5000
    
    instance = gLDA_nonCollapsed_lowDim_log(iters, path, result_path, simul_data)
    instance.sampling()
    
    #plot proportion of Topic Type 0
    true_prop = np.sum(simul_data['z']==0)/instance.total_words
    print("True proportion: ", true_prop)
    
    empri_prop = np.zeros((iters,1)).astype('float32')
    for iter_ in range(iters):
        empri_prop[iter_, 0] = np.sum(instance.zs[iter_, :,:] ==0)/instance.total_words
    
    fig, ax = plt.subplots()
    ax.plot(range(iters), empri_prop, '-b', label='Empirical Topic Type 0')
    leg = ax.legend()
    plt.title('Trajectory of Topic Type 0')
    plt.xlabel('Iters')
    plt.ylabel('Proportion of Topic Type 0')
    plt.savefig(result_path+'Trajectory of Topic Type 0')







