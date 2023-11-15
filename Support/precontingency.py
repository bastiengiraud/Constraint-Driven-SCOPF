# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 16:10:14 2017

@author: jlc516
"""

from __future__ import division

#import pandapower
import math
import numpy as np
import time
import torch


def compDisFac(net): # wordt gebruikt voor flagCont
    X = compX(net)
    d = compdf(X,net)
    return d 

def compX(net):
    #computes the matrix X: phi = X * P
    # phi = (B^-1) * P and (B^-1) = X
    #where phi are the phase angles and P are the injected power of the respective nodes    B = compB(net, ref)
    #print(ref)
    #print(net.bus)
    net["bus"].reset_index(drop=True,inplace = True)
    busmap = dict(zip(net["bus"]["name"],net["bus"].index)) 
    ref = busmap[net["par"]["refnode"][0]]
    B = compB(net,busmap,ref)
    X = invertB(B,ref)
    return X   

def compB(net,busmap,ref):
    #computes the matrix B: P = B * phi
    #where phi are the phase angles and P are the injected power of the respective nodes 
    B  = np.zeros( shape = (len(net["bus"].index), len(net["bus"].index)))
    for l in range(len(net["line"])):
        indexbfrom = busmap[net["line"].from_bus[l]]
        indexbto = busmap[net["line"].to_bus[l]]
        B[indexbfrom,indexbfrom] = B[indexbfrom,indexbfrom] + 1/(net["line"].x[l])
        B[indexbto,indexbto] = B[indexbto,indexbto] + 1/(net["line"].x[l])
        B[indexbfrom,indexbto] = B[indexbfrom,indexbto] - 1/(net["line"].x[l])
        B[indexbto,indexbfrom] = B[indexbto,indexbfrom] - 1/(net["line"].x[l])
    B[:,ref] = 0
    B[ref,:] = 0
    return B


def invertB(B,ref): # how computationally efficient are these inverses??
    #do matrix inversion but first deletes 
    B = np.delete(B, ref, axis=0) # delete slack bus
    B = np.delete(B, ref, axis=1) 
    X = np.linalg.pinv(B)
    X = np.insert(X, ref, values=0, axis=0) # add zero at position of slack bus
    X = np.insert(X, ref, values=0, axis=1)
    return X

def compptdfs(net):
    #https://www.tandfonline.com/doi/pdf/10.5370/JICEE.2011.1.1.049

    net["bus"].reset_index(drop=True,inplace = True)
    ref = net["par"]["refnode"][0]
    busmap = dict(zip(net["bus"]["name"],net["bus"].index)) 
    B = compB(net,busmap,ref) # B is essentially the jacobian
    X = invertB(B,ref) # X is inverse of B matrix  
    
    Bbr = np.diag(1/net["line"]['x']) # nline x nline where diagonal values are line reactances
    A = np.zeros( shape = (len(net["line"]), len(net["bus"])) ) # A is the from-to bus matrix
    # this the branch bus incidence matrix with 1's and -1's !!
    for i in range(len(net["line"])):
        A[i,net["line"]['from_bus'][i]] = -1
        A[i,net["line"]['to_bus'][i]] = 1
    PTDF = np.dot(np.dot(Bbr,A),X)
    return PTDF

def compptdfs_alt(net):
    #https://www.tandfonline.com/doi/pdf/10.5370/JICEE.2011.1.1.049

    net["bus"].reset_index(drop=True,inplace = True)
    ref = net["par"]["refnode"][0]
    busmap = dict(zip(net["bus"]["name"],net["bus"].index)) 
    B = compB(net,busmap,ref) # B is essentially the jacobian
    X = invertB(B,ref) # X is inverse of B matrix  
    
    Bbr = np.diag(1/net["line"]['x']) # nline x nline where diagonal values are line reactances
    A = np.zeros( shape = (len(net["line"]), len(net["bus"])) ) # A is the from-to bus matrix
    # this the branch bus incidence matrix with 1's and -1's !!
    for i in range(len(net["line"])):
        A[i,net["line"]['from_bus'][i]] = -1
        A[i,net["line"]['to_bus'][i]] = 1
    PTDF = np.dot(np.dot(np.dot(Bbr,A),X),np.transpose(A)) 
    return PTDF


def compdf(X,net): # copmute delta flow? difference in flow is difference in P x PTDF
    ref = net["par"]["refnode"][0]
    d = np.zeros( shape = (len(net["line"]), len(net["line"])) )
    net["bus"].reset_index(drop=True,inplace = True) 
    busmap = dict(zip(net["bus"].name,net["bus"].index))
    for l in range(len(net["line"])):
        for k in range(len(net["line"])):
            i = net["line"]['from_bus'][l]
            j = net["line"]['to_bus'][l]
            n = net["line"]['from_bus'][k]
            m = net["line"]['to_bus'][k]            
            ibi= busmap[net["line"]['from_bus'][l]]
            ibj= busmap[net["line"]['to_bus'][l]]
            ibn= busmap[net["line"]['from_bus'][k]]
            ibm= busmap[net["line"]['to_bus'][k]]
            xk = net["line"]['x'][k]
            xl = net["line"]['x'][l]
            if m == ref:
                delta_inm = np.divide(X[ibi][ibn]*xk,(X[ibn][ibm]-X[ibn][ibn]))
                delta_jnm = np.divide(X[ibj][ibn]*xk,(X[ibn][ibm]-X[ibn][ibn]))
            elif n == ref:
                delta_inm = - np.divide(X[ibi][ibm]*xk,(X[ibn][ibm]-X[ibm][ibm]))
                delta_jnm = - np.divide(X[ibj][ibm]*xk,(X[ibn][ibm]-X[ibm][ibm]))
            elif i == ref:
                delta_inm = 0
                delta_jnm = np.divide((X[ibj][ibn]-X[ibj][ibm])*xk,(xk-(X[ibn][ibn]+X[ibm][ibm]-2*X[ibn][ibm])))
            elif j == ref:
                delta_inm = np.divide((X[ibi][ibn]-X[ibi][ibm])*xk,(xk-(X[ibn][ibn]+X[ibm][ibm]-2*X[ibn][ibm])))
                delta_jnm = 0
            else:
                delta_inm = np.divide((X[ibi][ibn]-X[ibi][ibm])*xk,(xk-(X[ibn][ibn]+X[ibm][ibm]-2*X[ibn][ibm])))
                delta_jnm = np.divide((X[ibj][ibn]-X[ibj][ibm])*xk,(xk-(X[ibn][ibn]+X[ibm][ibm]-2*X[ibn][ibm])))
            if np.isinf(np.abs(delta_jnm)): delta_jnm = 0                
            if np.isinf(np.abs(delta_inm)): delta_inm = 0            
            d[l][k] =  1/xl * (np.nan_to_num(delta_inm) - np.nan_to_num(delta_jnm))
                #print(l, k )
                #print(d[l][k]=0 )
                #print(d[l][k] )            
    return d

def flagCont(net, linei, linecontingencies, d=None):
    if d == None: d = compDisFac(net)        
    newlinef = np.repeat(linei[:,np.newaxis], len(linei), 1) + np.multiply(d,np.repeat(linei[np.newaxis,:], len(linei), 0) )    
    max_f = np.repeat(net["line"].max_f[:,np.newaxis], len(linei), 1)
    #checks if all lines are in range = True 
    c = abs(newlinef)<= max_f
    #if cont_flag=False -> it is critical!
    cont_flag = np.ones(len(linecontingencies), dtype = 'bool')
    
    for i in linecontingencies:    
        if newlinef[i,i] == 0 or abs(newlinef[i,i])>10000:
            cont_flag[i] = 0
        c[i,i] = True
        if c[:,i].all() == False : cont_flag[i] = 0
        newlinef[i][i] = 0
    return newlinef, cont_flag



def flow_incidence(net, case, flow, batch_size):
    l = len(net["line"].index)
    A = torch.tensor(range(0, l))
    zero_tensor = torch.zeros(size = (1,batch_size), dtype = torch.float64)

    if case == 'N-1':
        Ncontingencies = int(l)
        A_baseflow = (A.unsqueeze(1)).T
        Fli0_train_diag = torch.zeros(size = (int(Ncontingencies),int(Ncontingencies),int(batch_size)),dtype = torch.float64)
        Fli0_train_tot = torch.diag_embed(flow, dim1 = 0, dim2 = 1)

    elif case == 'N-2':
        Ncontingencies = int(l*(l-1)/2)
        A_baseflow = torch.zeros(size = (2,Ncontingencies),dtype = torch.float64)
        Fli0_train_diag_index = torch.zeros(size = (int(Ncontingencies*2),int(Ncontingencies)),dtype = torch.float64)
    
        # define repeat pattern
        repeats = torch.tensor(list(reversed(range(0, l))))
    
        # define first two rows and stack
        A_baseflow_first = torch.repeat_interleave(A, repeats)
        A_baseflow_second = torch.cat([A[i+1:] for i in range(len(A))], dim=0)
        A_baseflow = torch.stack((A_baseflow_first.unsqueeze(1), A_baseflow_second.unsqueeze(1)), dim = 2)[:,0].T
    
        for i in range(int(Ncontingencies)):
            Fli0_train_diag_index[2+2*i-2,i] = A_baseflow[0,i]+1
            Fli0_train_diag_index[2+2*i-1,i] = A_baseflow[1,i]+1
        Fli0_train_diag_index = Fli0_train_diag_index.long()
        Fli0_train_diag_index_batches = Fli0_train_diag_index.T.unsqueeze(2).repeat(1,1,batch_size)
        Fli0_train_diag_index_flat = (Fli0_train_diag_index_batches.flatten(start_dim = 0, end_dim = 1))
        
        Flow = torch.cat((zero_tensor,flow.T))
        Flow = Flow
                
        Fli0_train_intermediate = (Flow[Fli0_train_diag_index_flat])
        Fli0_train_tot_flat = (Fli0_train_intermediate[:,0,:])
        Fli0_train_tot = Fli0_train_tot_flat.reshape(int(Ncontingencies),int(Ncontingencies*2),batch_size).permute(1,0,2)
                
        
        
    elif case == 'N-3':
        Ncontingencies = int(l*(l-1)/2*(l-2)/3)
        A_baseflow = torch.zeros(size = (3,Ncontingencies),dtype = torch.float64)
        
        # define repeat pattern
        sum_line = torch.tensor(np.sum(range(0,l)))
        repeats = torch.cat([(sum_line - l*i) for i in range(l)])
        #repeats1 = torch.tensor(list(reversed(range(int(sum_line-l-1), int(sum_line)))))
        
        # 496, 465, 435, 406 .....
    
        # define first three rows and stack
        #A_baseflow_first = torch.repeat_interleave(A, repeats1)
        #A_baseflow_second = torch.cat([A[i+1:] for i in range(len(A))], dim=0)
        #A_baseflow_third = []
        #A_baseflow = torch.stack((A_baseflow_first.unsqueeze(1), A_baseflow_second.unsqueeze(1), 
                                  #A_baseflow_third.unsqueeze(1)), dim = 2)[:,0].T
                                  
        # for i in range(int(Ncontingencies)):
        #     Fli0_train_diag_index[2+2*i-2,i] = A_baseflow[0,i]+1
        #     Fli0_train_diag_index[2+2*i-1,i] = A_baseflow[1,i]+1
    
    return Fli0_train_tot



def cont_flow(net, case, lodf, A, flow): # A = incidence matrix
    l = len(net["line"].index)
    l_index = torch.tensor(range(0,l))
    
    if case == 'N-1':
        l_cont = torch.tensor(range(0,l))
        flow_cont_all = torch.zeros(size = (l,l), dtype = torch.float64)
        
        flow_cont = map(lambda n: list(map(lambda i: lodf[n,A[0,i]]*flow[i], l_cont)), l_index)
        flow_cont = torch.tensor(list(flow_cont))
        flow_cont_all = flow_cont
            
        flow_cont_all = flow_cont_all + flow #broadcast
        
    if case == 'N-2':
        l_cont = torch.tensor(range(0,int(l*(l-1)/2)))
        flow_cont_all = torch.zeros(size = (l,int(l*(l-1)/2)), dtype = torch.float64)
        
        flow_cont = map(lambda n: list(map(lambda i: lodf[n,A[0,i]]*flow[A[0,i]] + lodf[n,A[1,i]]*flow[A[1,i]], l_cont)), l_index)
        flow_cont = torch.tensor(list(flow_cont))
        flow_cont_all = flow_cont
    
        #flow_cont_all = flow_cont_all + flow
        
    return flow_cont_all
















"""

def compptdfMO(net, i, j, k, case):

    net["bus"].reset_index(drop=True,inplace = True)
    ref = net["par"]["refnode"][0]
    busmap = dict(zip(net["bus"]["name"],net["bus"].index)) 
    B = compB(net,busmap,ref) # B is essentially the jacobian
    X = invertB(B,ref) # X is inverse of B matrix  
    
    Xm = np.diag(1/net["line"]['x']) # nline x nline where diagonal values are line reactances
    phi = np.zeros( shape = (len(net["line"]), len(net["bus"])) ) 
    for z in range(len(net["line"])):
        phi[z,net["line"]['from_bus'][z]] = -1
        phi[z,net["line"]['to_bus'][z]] = 1
    if case == 'N-1':
        psi = phi[i,:]
        psi_transpose = np.transpose(psi)
    elif case == 'N-2':
        psi1 = phi[i,:]
        psi2 = phi[j,:]
        psi = np.vstack((psi1,psi2))
        psi_transpose = np.transpose(psi)
    elif case == 'N-3':
        psi1 = phi[i,:]
        psi2 = phi[j,:]
        psi3 = phi[k,:]
        psi = np.vstack((psi1,psi2,psi3))
        psi_transpose = np.transpose(psi)
    PTDF = np.dot(np.dot(np.dot(Xm,phi),X),psi_transpose)
    return PTDF# , psi


def compptdfOO(net, i, j, k, case):

    net["bus"].reset_index(drop=True,inplace = True)
    ref = net["par"]["refnode"][0]
    busmap = dict(zip(net["bus"]["name"],net["bus"].index)) 
    B = compB(net,busmap,ref) # B is essentially the jacobian
    X = invertB(B,ref) # X is inverse of B matrix  
    
    if case == 'N-1':
        Xo = np.array(1/net["line"]['x'][i])
    elif case == 'N-2':
        Xo1 = np.array(net["line"]['x'][i]) 
        Xo2 = np.array(net["line"]['x'][j])
        Xo = np.array([Xo1,Xo2])
        Xo = np.linalg.inv(np.diag(Xo))
    elif case == 'N-3':
        Xo1 = np.array(net["line"]['x'][i]) 
        Xo2 = np.array(net["line"]['x'][j])
        Xo3 = np.array(net["line"]['x'][k])
        Xo = np.array([Xo1,Xo2,Xo3])
        Xo = np.linalg.inv(np.diag(Xo))
    
    phi = np.zeros( shape = (len(net["line"]), len(net["bus"])) ) 
    for z in range(len(net["line"])):
        phi[z,net["line"]['from_bus'][z]] = -1
        phi[z,net["line"]['to_bus'][z]] = 1
    if case == 'N-1':
        psi = phi[i,:]
        psi_transpose = np.transpose(psi)
    elif case == 'N-2':
        psi1 = phi[i,:]
        psi2 = phi[j,:]
        psi = np.vstack((psi1,psi2))
        psi_transpose = np.transpose(psi)
    elif case == 'N-3':
        psi1 = phi[i,:]
        psi2 = phi[j,:]
        psi3 = phi[k,:]
        psi = np.vstack((psi1,psi2,psi3))
        psi_transpose = np.transpose(psi)
    PTDF = np.dot(np.dot(np.dot(Xo,psi),X),psi_transpose)
    return PTDF




"""