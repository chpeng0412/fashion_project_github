#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 10:26:28 2021

@author: chloepeng
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import random
import scipy.special as ss
import time
from sklearn.cluster import KMeans
import pandas as pd


class Gauss_LDA():
    def __init__(self, iters, num_topic, kappa, user_look_file, user_look_date_file, embeddings_file):
        #load data
        self.data_path = "../../data/processed_real_data/"
        self.result_path = "../../results/collapsed_results/"
        self.doc_map_data = np.load(self.data_path+user_look_file)
        self.user_look_date = np.load(self.data_path+user_look_date_file, allow_pickle=True)
        self.v = np.load(self.data_path+embeddings_file)


        #set up parameters
        self.total_words=self.v.shape[0]
        self.num_embed = self.v.shape[1]
        self.users, self.user_counts= np.unique(self.doc_map_data[:,0], return_counts = True)
        self.num_doc = self.users.shape[0]
        self.docs = np.repeat(np.array([doc for doc in range(self.num_doc)]), self.user_counts).reshape((self.total_words, 1))
        self.doc_map = np.concatenate((self.docs, self.doc_map_data), axis =1)
        self.num_topic = num_topic
        
        #set up priors
        self.alpha = np.array(np.random.uniform(low= 10, high = 20, size = num_topic))
        self.kappa= kappa
        self.nu = self.num_embed+2
        
        #setting priors mu_0 and psi
        kmeans =KMeans(n_clusters = self.num_topic).fit(self.v)
        labels = kmeans.labels_
        sigmas = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        mus = np.zeros((self.num_topic, self.num_embed))
        for topic in range(self.num_topic):
            mus[topic, :]= np.mean(self.v[labels==topic, :], axis =0)
            sigmas[topic, :, :] = np.cov(self.v[labels==topic, :].T)*(self.v[labels==topic, :].shape[0]/self.total_words)
            
        self.mu_0 = np.mean(mus, axis =0)
        sigma_mean = np.mean(sigmas)+np.identity(self.num_embed)
        self.psi = np.linalg.inv(sigma_mean*(1/(self.nu+2)))

       #initialization
        self.iters = iters
        self.zs = np.zeros((self.iters, self.total_words, 1))
        self.Ls_first = np.ones((self.iters, self.num_topic, 1,1))*1e-10
        self.muks = np.zeros((self.iters, self.num_topic, self.num_embed))
        self.theta_hat = None
        lst = [topic for topic in range(self.num_topic)]
        self.zs[0, :, 0] = random.choices(lst, k = self.total_words)
        
        
    def Nk_func(self, i):
        """
        measures the count of each topics. The sum of Nk is equal to total_words
        """
        values, Nk1 = np.unique(self.zs[i, :, 0], return_counts = True)
        Nk = np.zeros((self.num_topic,))
        for topic in range(self.num_topic):
            if topic in values:
                Nk[topic] = Nk1[np.argwhere(values==topic).item()]
            else:
                Nk[topic] = 0
        return Nk
    
    def vk_hat_func( self, i, Nk):
        """
        measures the averaged embeddings of each topic
        """
        for topic in range(self.num_topic):
            vdk = np.sum(self.v[self.zs[i,:,0]==topic], axis =0)
            if Nk[topic] !=0:
                vk_hat = vdk/Nk[topic]
            else:
                vk_hat = 0
        return vk_hat
        

    def muks_func(self,i, Nk, vk_hat):
        """
        measures the mu of each topic
        """
        
        muk = np.zeros((self.num_topic, self.num_embed))
        for topic in range(self.num_topic):
            muk[topic,:] = (self.kappa*self.mu_0 + Nk[topic]*vk_hat)/(self.kappa+Nk[topic])
        return muk
    
    def sigmak_func(self,i, Nk, vk_hat):
        """
        measures the sigma of each topic

        """
        sigmak = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        for topic in range(self.num_topic):
            diff= self.v[self.zs[i,:,0]==topic] - vk_hat
            Ck = diff.T @diff        
            mu_diff = vk_hat - self.mu_0
            quar = (mu_diff).reshape(self.num_embed,1) @(mu_diff).reshape(1,self.num_embed)
            psi_second = ((self.kappa*Nk[topic])/(self.kappa+Nk[topic]))*quar        
            psi_k = self.psi +Ck + psi_second
            sigmak[topic, :, :]= psi_k/(self.nu + Nk[topic]-self.num_embed +1)
        return sigmak
        
    def ndk_func(self,i):
        """
        counts the numberof topic for each doc
        """
        ndk = np.zeros((self.num_doc, self.num_topic))
        for doc in range(self.num_doc):
            values, ndk_1 = np.unique(self.zs[i, :, 0][self.doc_map[:,0]==doc], return_counts = True)
            ndk_1_new =np.zeros((1, self.num_topic))
            for topic in range(self.num_topic):
                if topic in values:
                    ndk_1_new[0, topic] = ndk_1[np.argwhere(values==topic).item()]
                else:
                    ndk_1_new[0,topic] =0
            ndk[doc, :] = ndk_1_new
        return ndk
        
    def log_det_func(self, i, sigmak):
        """
        calculate the logrithmic determinant of each topic varaince-covariance matrix

        """
        log_det = np.zeros((self.num_topic, 1))
        for topic in range(self.num_topic):
            sigma_k = sigmak[topic,:,:]
            L = np.linalg.cholesky(sigma_k)
            diag = np.diag(L)
            log_det[topic, 0]= 2*np.sum(np.log(diag))   
        return log_det
    
    def inv_sigma_func(self, sigmak):
        """
        measures the inverse of sigma for each topic

        """
        inv_sigma = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        for topic in range(self.num_topic):
            sigma_k = sigmak[topic,:,:]
            inv_sigma[topic, :, :] = np.linalg.inv(sigma_k) 
        return inv_sigma
    
    def L_new_func(self,i, sigmak):
        """
        measures the Cholesky decomposition of sigma of each topic

        """
        L_new = np.linalg.cholesky(sigmak)
        return L_new
       
                
    def zs_func(self,i, ndk, L_new, Nk,muk):
        """
        draw topic assignments for all of the words

        """
        docs, doc_counts = np.unique(self.doc_map[:,0], return_counts = True)
        ndk_long = np.repeat(ndk, doc_counts, axis =0)

        L_inv = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        log_det = np.zeros((self.num_topic, 1))
        for topic in range(self.num_topic):
            diag = np.diag(L_new[topic, :, :])
            log_det[topic, 0]= 2*np.sum(np.log(diag))
            L_inv[topic, :, :] = np.linalg.inv(L_new[topic, :, :]) 
    
        log_det_long = np.repeat(log_det.reshape(1,self.num_topic),self.total_words, axis =0)
        ndk_alpha_long = ndk_long+self.alpha
        Nk_long = np.repeat(Nk.reshape(1, self.num_topic), self.total_words, axis =0)
        nuk_long = self.nu+Nk_long
        gamma_nup = ss.gammaln((nuk_long+self.num_embed)/2)
        gamma_nu = ss.gammaln(nuk_long/2)
        
        v_expand = np.expand_dims(self.v, axis = 1)
        v_long = np.repeat(v_expand, self.num_topic, axis =1)
        b_long = v_long-muk
      
        L_inv_b = np.zeros((self.num_topic, self.total_words, self.num_embed))
        for topic in range(self.num_topic):
            L_inv_b[topic, :, :] = np.matmul(L_inv[topic,:, :], b_long[:, topic, :].T).T
        
        L_inv_b_twice = np.zeros((self.total_words, self.num_topic))
        for topic in range(self.num_topic):
            L_inv_b_twice[:, topic] = np.sum(L_inv_b[topic,:, :]**2, axis = 1)
        quadratic_long = np.log(1+np.multiply(1/nuk_long, L_inv_b_twice))
    
        
        log_1_long = np.log(ndk_alpha_long)
        z_log_long = log_1_long + gamma_nup -gamma_nu -(self.num_embed/2)*np.log(nuk_long)-0.5*log_det_long - np.multiply( (nuk_long+self.num_embed)/2, quadratic_long)-(self.num_embed/2)*np.pi
        z_log_max = np.max(z_log_long, axis =1)
        z_log_new = z_log_long - z_log_max.reshape(self.total_words, 1)
        z_nonlog = np.exp(z_log_new)
        z_non_log_norm = z_nonlog/np.sum(z_nonlog, axis =1).reshape(self.total_words, 1)
        
        sum_bigger_1_idx = np.where(np.sum(z_non_log_norm, axis =1)>1)[0]
        
        if len(sum_bigger_1_idx) >0:
            for idx in sum_bigger_1_idx:
                max_index = np.argwhere(z_non_log_norm[idx, :] == np.max(z_non_log_norm[idx, :])).item()
                z_non_log_norm[idx,max_index] =  z_non_log_norm[idx,max_index]  -1e-5    
        for word in range(self.total_words):
            self.zs[i+1, word, 0] = np.argmax(sp.multinomial.rvs(1,z_non_log_norm[word,:]))

        empri_prop = np.sum(self.zs[i+1, :, :] ==0)/self.total_words
        print(i+1, empri_prop)
    
        
    def sampling(self):
        """
        do self.iters round of sampling
        """
        Nk = self.Nk_func(0)
        vk_hat = self.vk_hat_func(0, Nk)
        muk = self.muks_func(0, Nk, vk_hat)
        self.muks[0, :, :] =muk
        sigmak = self.sigmak_func(0, Nk, vk_hat)
        inv_sigma = self.inv_sigma_func(sigmak)

        ndk = self.ndk_func(0)        
        log_det = self.log_det_func(0, sigmak)
        L_new = self.L_new_func(0, sigmak)
        self.Ls_first[0, :, 0, 0]=L_new[:, 0, 0]

        
        for word in range(self.total_words):
            zk_log = np.zeros((1,self.num_topic))
            for topic in range(self.num_topic):
                doc_idx = int(self.doc_map[:,0][word])
                log_1 = np.log(ndk[doc_idx, topic] + self.alpha[topic])
                nuk = self.nu + Nk[topic]
                gamma_nup = ss.gammaln((nuk+self.num_embed)/2)
                gamma_nu = ss.gammaln(nuk/2)
                quadratic =np.log(1+ (1/nuk)* np.matmul(np.matmul((self.v[word] -muk[topic, :]).reshape(1,self.num_embed), inv_sigma[topic,:,:]), (self.v[word] -muk[topic, :]).reshape(self.num_embed,1)))
                zk_log[0,topic] = log_1 + gamma_nup-gamma_nu -(self.num_embed/2)*np.log(nuk)-0.5*log_det[topic,0]-((nuk+self.num_embed)/2)*quadratic
            zk_log_max = np.max(zk_log[0, :])
            zk_log_new = zk_log-zk_log_max
            zk_nonlog=np.exp(zk_log_new)
            zk_non_log_norm = zk_nonlog/np.sum(zk_nonlog)
            if np.sum(zk_non_log_norm)>1:
                diff = np.sum(zk_non_log_norm)-1
                max_index = np.argwhere(zk_non_log_norm[0, :] == np.max(zk_non_log_norm)).item()
                zk_non_log_norm[0,max_index] =  zk_non_log_norm[0,max_index]  -1e-5    
            self.zs[1, word, 0] = np.argmax(sp.multinomial.rvs(1,zk_non_log_norm[0,:]))
        
        for i in range(1, self.iters -1):
            Nk = self.Nk_func(i)
            vk_hat = self.vk_hat_func(i, Nk)
            muk = self.muks_func(i, Nk, vk_hat)
            self.muks[i, :, :] = muk
            sigmak = self.sigmak_func(i, Nk, vk_hat)
            L_new= self.L_new_func(i, sigmak)
            self.Ls_first[i, :, 0, 0]=L_new[:, 0, 0]
                       
            ndk = self.ndk_func(i)
            self.theta_hat = ndk/np.sum(ndk, axis = 1).reshape(self.num_doc, 1)
            topic_prop = np.sum(self.theta_hat, axis =0)/np.sum(self.theta_hat)
            
            self.zs_func(i, ndk, L_new, Nk, muk)
            

