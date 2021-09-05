#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 26 14:23:58 2021

@author: chloepeng
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import random
import scipy.special as ss
import helper 
import time

class gLDA_Collapsed_Speed:
    """
    Collapsed gLDA with Rank One Update. However, this algorithm requires to store the previous inverse varaince-covaraince matrix, which takes a lot of space in high dimensional case. 
    It is better to use the vectorized version called gLDA_Collapsed_Speed_vector in this folder. 
    """
    def __init__(self, iters, path, results_path, simul_data):
        self.iters = iters
        self.path = path
        self.simul_data= simul_data
        
        #prior 
        self.num_topic = self.simul_data['num_topic']
        self.num_embed = self.simul_data['num_embed']
        self.alpha = self.simul_data['alpha']
        self.mu_0 = self.simul_data['mu_0']
        self.nu = self.num_embed+2
        self.psi = self.simul_data['psi']
        self.kappa = self.simul_data['kappa']
        
        #initialization
        self.num_doc = self.simul_data['num_doc']
        self.total_words = self.simul_data['total_words']
        self.doc_map = self.simul_data['doc_map']
        self.v = self.simul_data['v']
        true_prop =np.sum(self.simul_data['z']==0)/self.total_words
        print('true_prop', true_prop)
    
    
        self.iters =iters
        self.zs = np.zeros((self.iters, self.total_words, 1))
        self.Ls = np.ones((self.iters, self.num_topic, self.num_embed, self.num_embed))*1e-100
        self.muks = np.zeros((self.iters, self.num_topic, self.num_embed))
        self.Nks =np.zeros((self.iters, self.num_topic))
    
    
        lst = [topic for topic in range(self.num_topic)]
        self.zs[0,:,0] =random.choices(lst, k = self.total_words)
    
    def Nk_func(self,i):
        values, Nk1 = np.unique(self.zs[i, :, 0], return_counts = True)
        Nk = np.zeros((self.num_topic,))
        for topic in range(self.num_topic):
            if topic in values:
                Nk[topic] = Nk1[np.argwhere(values==topic).item()]
            else:
                Nk[topic] = 0  
        return Nk
    
    def muk_func(self,i, Nk):
        muk = np.zeros((self.num_topic, self.num_embed))
        for topic in range(self.num_topic):
            vdk = np.sum(self.v[self.zs[i,:,0]==topic], axis =0)
            vk_hat = vdk/Nk[topic]
            muk[topic,:] = (self.kappa*self.mu_0 + Nk[topic]*vk_hat)/(self.kappa+Nk[topic])
        return muk
    
    
    def Ls_rank_one(self,i):
        diff_idx = np.where(self.zs[i, :, :] != self.zs[i-1, :, :])
        L_new = self.Ls[i-1,:, :, :].copy()
        for word in diff_idx[0]:
            new_topic = int(self.zs[i, word, 0])
            old_topic = int(self.zs[i-1, word, 0])
            
            b_new_topic = self.muks[i,new_topic, :] - self.v[word]
            old_idx = i-1
            while self.zs[old_idx, word, 0] ==self.zs[old_idx-1, word, 0]:
                old_idx -=1
                if old_idx-1 <0:
                    old_idx = 0
                    break
                
            # print(old_idx)
            b_old_topic = self.muks[old_idx,old_topic, :]-self.v[word]
            
            kappa_k_new_topic = self.kappa + self.Nks[i,new_topic]
            kappa_k_old_topic = self.kappa + self.Nks[old_idx, old_topic]
            
            nu_new_topic = self.nu +self.Nks[i,new_topic]-self.num_embed + 1
            nu_old_topic = self.nu+self.Nks[old_idx, old_topic]-self.num_embed + 1
            
            y_new_topic = np.sqrt(kappa_k_new_topic/(kappa_k_new_topic-1))*np.sqrt(1/nu_new_topic)*b_new_topic
            y_old_topic = np.sqrt(kappa_k_old_topic/(kappa_k_old_topic-1))*np.sqrt(1/nu_old_topic)*b_old_topic
         
            L_new[new_topic, :, :] = helper.cholupdate(L_new[new_topic, :,:].T, y_new_topic, "+")
            L_new[old_topic, :, :] = helper.cholupdate(L_new[old_topic, :, :].T, y_old_topic, "-")  
            
        return L_new

    def sigmak_func(self,i, Nk):
        # muk = np.zeros((num_topic, num_embed))
        sigmak = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        # Lk = np.zeros((num_topic, num_embed, num_embed))
        for topic in range(self.num_topic):
            vdk = np.sum(self.v[self.zs[i,:,0]==topic], axis =0)
            vk_hat = vdk/Nk[topic]
            
            diff= self.v[self.zs[i,:,0]==topic] - vk_hat
            Ck = diff.T @diff        
            mu_diff = vk_hat - self.mu_0
            quar = (mu_diff).reshape(self.num_embed,1) @(mu_diff).reshape(1,self.num_embed)
            psi_second = ((self.kappa*Nk[topic])/(self.kappa+Nk[topic]))*quar        
            psi_k = self.psi +Ck + psi_second
            sigmak[topic, :, :]= psi_k/(self.nu + Nk[topic]-self.num_embed +1)
        return sigmak
    
    def ndk_func(self,i):
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
    
    def sample_z(self,i, ndk, Nk, muk):
        docs, doc_counts = np.unique(self.doc_map[:,0], return_counts = True)
        ndk_long = np.repeat(ndk, doc_counts, axis =0)
        
        L_inv = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        log_det = np.zeros((self.num_topic, 1))
        for topic in range(self.num_topic):
            diag = np.diag(self.Ls[i, topic, :, :])
            log_det[topic, 0]= 2*np.sum(np.log(diag))
            L_inv[topic, :, :] = np.linalg.inv(self.Ls[i, topic, :, :])
        
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
        z_log_long = log_1_long + gamma_nup -gamma_nu -(self.num_embed/2)*np.log(nuk_long)-0.5*log_det_long - np.multiply( (nuk_long+self.num_embed)/2, quadratic_long)
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

    
    def sampling(self):
        Nk = self.Nk_func(0)
        muk = self.muk_func(0, Nk)
        self.muks[0,:, :] = muk
        sigmak = self.sigmak_func(0, Nk)
        
        ndk = np.zeros((self.num_doc, self.num_topic))
        for doc in range(self.num_doc):
            values, ndk_1 = np.unique(self.zs[0, :, 0][self.doc_map[:,0]==doc], return_counts = True)
            ndk_1_new =np.zeros((1, self.num_topic))
            for topic in range(self.num_topic):
                if topic in values:
                    ndk_1_new[0, topic] = ndk_1[np.argwhere(values==topic).item()]
                else:
                    ndk_1_new[0,topic] =0
            ndk[doc, :] = ndk_1_new
            
        inv_sigma = np.zeros((self.num_topic, self.num_embed, self.num_embed))
        log_det = np.zeros((self.num_topic, 1))
        for topic in range(self.num_topic):
            sigma_k = sigmak[topic,:,:]
            # using psedo-inverse here b
            inv_sigma[topic, :, :] = np.linalg.inv(sigma_k)        
            L = np.linalg.cholesky(sigma_k)
            diag = np.diag(L)
            log_det[topic, 0]= 2*np.sum(np.log(diag))   

        #calculate the first L using updating
        for word in range(self.total_words):
            new_topic = int(self.zs[0, word, 0])
            b_new_topic = self.muks[0, new_topic, :]-self.v[word]
            kappa_k_new_topic = self.kappa + self.Nks[0, new_topic]
            nu_new_topic = self.nu + self.Nks[0, new_topic]-self.num_embed +1
            y_new_topic = np.sqrt(kappa_k_new_topic/(kappa_k_new_topic-1))*np.sqrt(1/nu_new_topic)*b_new_topic
            self.Ls[0, new_topic, :, :]= helper.cholupdate(self.Ls[0,new_topic, :,:].T, y_new_topic, "+")
        
        for word in range(self.total_words):
            zk = np.zeros((1,self.num_topic))
            for topic in range(self.num_topic):
                #calculate pdf in log space
                doc_idx = self.doc_map[:,0][word]
                log_1 = np.log(ndk[doc_idx, topic] + self.alpha[topic])
                nuk = self.nu + Nk[topic]
                gamma_nup = ss.gammaln((nuk+self.num_embed)/2)
                gamma_nu = ss.gammaln(nuk/2)
                quadratic =np.log(1+ (1/nuk)* np.matmul(np.matmul((self.v[word] -muk[topic, :]).reshape(1,self.num_embed), inv_sigma[topic,:,:]), (self.v[word] -muk[topic, :]).reshape(self.num_embed,1)))
                zk[0,topic] = np.exp(log_1 + gamma_nup-gamma_nu -(self.num_embed/2)*np.log(nuk)-0.5*log_det[topic,0]-((nuk+self.num_embed)/2)*quadratic)
            zk = zk/np.sum(zk)
            self.zs[1, word, 0] = np.argmax(sp.multinomial.rvs(1,zk[0,:]))   
            

        for i in range(1, self.iters-1):
            start = time.time()
            Nk=self.Nk_func(i)
            self.Nks[i, :] = Nk
            muk = self.muk_func(i, Nk)
            self.muks[i, :, :] = muk
            L_new = self.Ls_rank_one(i)
            self.Ls[i, :, :, :]=L_new
            
            sigmak = self.sigmak_func(i,Nk)
            ndk = self.ndk_func(i)
 
            self.sample_z(i, ndk, Nk, muk)
            end = time.time()
            empri_prop = np.sum(self.zs[i+1, :, :] ==0)/self.total_words
            print(i+1, empri_prop, np.round(end-start, 2))
        
    
if __name__=='main':
    path = "../../data/simul_data/"
    result_path = "../../results/collapsed_results/"
    simul_data = np.load(path+'simul_3_k_2_doc_300_kappa_5.npy', allow_pickle = True).item()
    iters = 200
    instance = gLDA_Collapsed_Speed(iters, path, result_path, simul_data)
    instance.sampling()

    empri_prop = np.zeros((iters,1)).astype('float32')
    for iter_ in range(iters):
        empri_prop[iter_, 0] = np.sum(instance.zs[iter_, :,:] ==0)/instance.total_words
    
    #do the time testing on high dimension
    fig, ax = plt.subplots()
    ax.plot(range(iters), empri_prop, '-b', label='Empirical Topic Type 0')
    leg = ax.legend()
    plt.title('Trajectory of Topic Type 0')
    plt.xlabel('Iters')
    plt.ylabel('Proportion of Topic Type 0')
    plt.savefig(result_path+'collapsed_rank_one_update_dim'+str(instance.num_embed)+'_k_'+str(instance.num_embed)+ '_Trajectory_of_Topic_Type_0')

