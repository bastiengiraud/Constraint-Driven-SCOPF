import sys
sys.path.append(r'/home/bgiraud')

import cvxpy as cp
import pandas as pd
import numpy as np
import math
import cvxpy_dcopf as cd
import cvxpy_scopf as cs
#import cvxpy_imb as ci
import torch
import torchvision
import timeit
import precontingency as pc
import torch.nn as nn
import time

from tqdm import tqdm
import itertools

from torch import Tensor


data = pd.read_excel('IEEE118spyros.xlsx',sheet_name=None)
Sbase = data['par']['base'][0]
lcontingencies = range(len(data['line']))
Nloads = len(data['load'])
Ngens = len(data['gen'])
Nlines = len(data['line'])
Nbus = len(data['bus'])

Nsamples = 500
Niterations = 5
batch_size = 10 # IEEE118, for N-2: 25 goes okay
train = 0.8
tollerance = 20 # tollerance line violation
learning_rate = 0.01
wd = 0.1 
case = 'N-3'

from torch.nn.parallel import DataParallel

if torch.cuda.is_available():
    # GPU is available
    device = torch.device('cuda')
    print("GPU is available")
else:
    # GPU is not available
    device = torch.device('cpu')
    print("GPU is not available")


if torch.cuda.device_count() >= 1:
    device_ids = list(range(torch.cuda.device_count()))
else:
    device_ids = None
    
num_workers = torch.cuda.device_count()

torch.set_default_tensor_type(torch.DoubleTensor)
l = len(data['line'])

if case == 'N-2':
    N2 = np.array(list(itertools.combinations(range(l),2)))   

elif case == 'N-3':
    N3 = np.array(list(itertools.combinations(range(l),3)))
    
elif case == 'N-4':
    N4 = np.array(list(itertools.combinations(range(l),4)))
    
elif case == 'N-5':
    N4 = np.array(list(itertools.combinations(range(l),5)))

    
def compute_lodf_N1(k):
    LODF_sheet = torch.zeros((l,l))#.to(device)
    PTDF_MO = np.transpose(np.array([p[:,i]]))
    PTDF_MO = torch.from_numpy(PTDF_MO)
    PTDF_OO = torch.from_numpy(np.array(p[i,i]))
    if PTDF_OO > 0.9999:
        LODF_col = torch.zeros(size = (l,1)).to(device) # WRONG; should be just the values where PTDF = 1
        singular = 1
    else:
        one = torch.eye(1)
        RHS = (1/(one - PTDF_OO))#.to(device)
        LODF_col = torch.matmul(PTDF_MO,RHS)#.to(device)
        LODF_col[i,:] = 0 # if a line is outaged, there can be no flow over it. Set own LODF to zero
        singular = 0

    LODF_sheet[i,:] = LODF_col[:,0].T

    LODF_sheet[abs(LODF_sheet) < 1e-5] = 0
    LODF_sheet_coo = LODF_sheet.to_sparse_coo()#.to(device)
    
    return LODF_sheet_coo, singular
    
    
def compute_lodf_N2(k):
    i = N2[k,0]#N2[k][0]
    j = N2[k,1]#N2[k][1]
    LODF_sheet = torch.zeros((l,l))#.to(device)
    PTDF_MO = np.transpose(np.array([p[:,i],p[:,j]]))
    PTDF_MO = torch.from_numpy(PTDF_MO)
    PTDF_OO = torch.from_numpy(np.array([[p[i,i],p[i,j]],[p[j,i],p[j,j]]]))
    one = torch.eye(2).double()#.to(device)
    subtract = (one - PTDF_OO).double()
    if torch.linalg.det(subtract) > 1e-12:
        RHS = (torch.linalg.solve(subtract, one))#.to(device)
        LODF_col = torch.matmul(PTDF_MO, RHS)#.to(device)
        LODF_col[i,:] = 0
        LODF_col[j,:] = 0
        singular = 0
    else: 
        PTDF_OO[(PTDF_OO >= 0.9999) & (PTDF_OO < 1.00001)] = 1
        LODF_col = torch.zeros(size = (l,2))#.to(device)
        singular = 1
        
    LODF_sheet[i,:] = LODF_col[:,0].T
    LODF_sheet[j,:] = LODF_col[:,1].T
    
    LODF_sheet[abs(LODF_sheet) < 1e-5] = 0
    LODF_sheet_coo = LODF_sheet.to_sparse_coo()#.to(device)

    return LODF_sheet_coo, singular

def compute_lodf_N3(k):
    i = N3[k,0]
    j = N3[k,1]
    r = N3[k,2]
    LODF_sheet = torch.zeros((l,l))#.to(device)
    PTDF_MO = np.transpose(np.array([p[:,i],p[:,j],p[:,r]]))
    PTDF_MO = torch.from_numpy(PTDF_MO)
    PTDF_OO = torch.from_numpy(np.array([[p[i,i],p[i,j],p[i,r]],[p[j,i],p[j,j],p[j,r]],[p[r,i],p[r,j],p[r,r]]]))
    one = torch.eye(3)#.to(device)
    subtract = one - PTDF_OO
    if torch.linalg.det(subtract) > 1e-9: # 1e-12
        RHS = (torch.linalg.solve(subtract, one))#.to(device)
        LODF_col = torch.matmul(PTDF_MO,RHS)#.to(device)
        LODF_col[i,:] = 0
        LODF_col[j,:] = 0
        LODF_col[r,:] = 0
        singular = 0
    else: 
        PTDF_OO[(PTDF_OO >= 0.9999) & (PTDF_OO < 1.0001)] = 1
        LODF_col = torch.zeros(size = (l,3))#.to(device)
        singular = 1

    LODF_sheet[i,:] = LODF_col[:,0].T
    LODF_sheet[j,:] = LODF_col[:,1].T
    LODF_sheet[r,:] = LODF_col[:,2].T

    LODF_sheet[abs(LODF_sheet) < 1e-5] = 0
    LODF_sheet_coo = LODF_sheet.to_sparse_coo()#.to(device)

    return LODF_sheet_coo, singular

def compute_lodf_N4(k):
    i = N4[k,0]
    j = N4[k,1]
    r = N4[k,2]
    t = N4[k,3]
    LODF_sheet = torch.zeros((l,l))#.to(device)
    PTDF_MO = np.transpose(np.array([p[:,i],p[:,j],p[:,r],p[:,t]]))
    PTDF_MO = torch.from_numpy(PTDF_MO)
    PTDF_OO = torch.from_numpy(np.array([[p[i,i],p[i,j],p[i,r],p[i,t]],[p[j,i],p[j,j],p[j,r],p[j,t]],[p[r,i],p[r,j],p[r,r],p[r,t]],[p[t,i],p[t,j],p[t,r],p[t,t]]]))
    one = torch.eye(4)#.to(device)
    subtract = one - PTDF_OO
    if torch.linalg.det(subtract) > 1e-9: # 1e-12
        RHS = (torch.linalg.solve(subtract, one))#.to(device)
        LODF_col = torch.matmul(PTDF_MO,RHS)#.to(device)
        LODF_col[i,:] = 0
        LODF_col[j,:] = 0
        LODF_col[r,:] = 0
        LODF_col[t,:] = 0
        singular = 0
    else: 
        PTDF_OO[(PTDF_OO >= 0.9999) & (PTDF_OO < 1.0001)] = 1
        LODF_col = torch.zeros(size = (l,4))#.to(device)
        singular = 1

    LODF_sheet[i,:] = LODF_col[:,0].T
    LODF_sheet[j,:] = LODF_col[:,1].T
    LODF_sheet[r,:] = LODF_col[:,2].T
    LODF_sheet[t,:] = LODF_col[:,3].T

    LODF_sheet[abs(LODF_sheet) < 1e-5] = 0
    LODF_sheet_coo = LODF_sheet.to_sparse_coo()#.to(device)

    return LODF_sheet_coo, singular

def compute_lodf_N5(k):
    i = N4[k,0]
    j = N4[k,1]
    r = N4[k,2]
    t = N4[k,3]
    y = N4[k,4]
    LODF_sheet = torch.zeros((l,l))#.to(device)
    PTDF_MO = np.transpose(np.array([p[:,i],p[:,j],p[:,r],p[:,t],p[:,y]]))
    PTDF_MO = torch.from_numpy(PTDF_MO)
    PTDF_OO = torch.from_numpy(np.array([[p[i,i],p[i,j],p[i,r],p[i,t],p[i,y]],[p[j,i],p[j,j],p[j,r],p[j,t],p[j,y]],[p[r,i],p[r,j],p[r,r],p[r,t],p[r,y]],[p[t,i],p[t,j],p[t,r],p[t,t],p[t,y]],[p[y,i],p[y,j],p[y,r],p[y,t],p[y,y]]]))
    one = torch.eye(5)#.to(device)
    subtract = one - PTDF_OO
    if torch.linalg.det(subtract) > 1e-9: # 1e-12
        RHS = (torch.linalg.solve(subtract, one))#.to(device)
        LODF_col = torch.matmul(PTDF_MO,RHS)#.to(device)
        LODF_col[i,:] = 0
        LODF_col[j,:] = 0
        LODF_col[r,:] = 0
        LODF_col[t,:] = 0
        LODF_col[y,:] = 0
        singular = 0
    else: 
        PTDF_OO[(PTDF_OO >= 0.9999) & (PTDF_OO < 1.0001)] = 1
        LODF_col = torch.zeros(size = (l,5))#.to(device)
        singular = 1

    LODF_sheet[i,:] = LODF_col[:,0].T
    LODF_sheet[j,:] = LODF_col[:,1].T
    LODF_sheet[r,:] = LODF_col[:,2].T
    LODF_sheet[t,:] = LODF_col[:,3].T
    LODF_sheet[y,:] = LODF_col[:,4].T

    LODF_sheet[abs(LODF_sheet) < 1e-5] = 0
    LODF_sheet_coo = LODF_sheet.to_sparse_coo()#.to(device)

    return LODF_sheet_coo, singular
	
torch.set_default_tensor_type(torch.DoubleTensor)
    
p = pc.compptdfs_alt(data)

l = len(data['line'])
start_lodf = time.time()

if case == 'N-1':
    Ncontingencies = l
    n_combinations = np.arange(Ncontingencies)
        
elif case == 'N-2':
    Ncontingencies = int(l*(l-1)/2)
    combinations = list(itertools.combinations(range(l), 2))
    n_combinations = np.arange(Ncontingencies)
        
elif case == 'N-3':
    Ncontingencies = int(l*((l-1)/2)*((l-2)/3))
    combinations = list(itertools.combinations(range(l), 3))
    n_combinations = np.arange(Ncontingencies)
    
elif case == 'N-4':
    Ncontingencies = int(l*((l-1)/2)*((l-2)/3)*((l-3)/4))
    combinations = list(itertools.combinations(range(l), 4))
    n_combinations = np.arange(Ncontingencies)
    
elif case == 'N-5':
    Ncontingencies = int(l*((l-1)/2)*((l-2)/3)*((l-3)/4)*((l-4)/5))
    combinations = list(itertools.combinations(range(l), 5))
    n_combinations = np.arange(Ncontingencies)

if __name__ == '__main__':
    
    #==== initialize lodf 
    lodf_batch = 20000
    tot_lodf_batches = math.ceil(Ncontingencies/lodf_batch)
    LODF = []

    #================ version 2
    lodf_batch_num = 0
    counter = 0
    lodf_list = list(range(lodf_batch))
    singular_list = []

    for i in tqdm(range(Ncontingencies)):
        if case == 'N-1':
            result, singular = compute_lodf_N1(i)
        elif case == 'N-2':
            result, singular = compute_lodf_N2(i)
        elif case == 'N-3':
            result, singular = compute_lodf_N3(i)
        elif case == 'N-4':
            result, singular = compute_lodf_N4(i)
        elif case == 'N-5':
            result, singular = compute_lodf_N5(i)
        
        # Append result directly to lodf_list and increment counter
        lodf_list[counter] = result
        counter += 1
        if singular == 1:
            singular_list.append(i)

        if counter == lodf_batch:
            # Concatenate lodf_list tensors in-place
            lodf_concatenated = torch.cat(lodf_list, dim=1).to(torch.double).requires_grad_()
            indices1 = lodf_concatenated._indices()[0].to(torch.float32)
            indices2 = lodf_concatenated._indices()[1].to(torch.float32)
            values = lodf_concatenated._values().to(torch.float32) 

            # Store lodf_concatenated in LODF_dict and reset counter and lodf_list
            LODF.append((indices1, indices2, values))
            counter = 0
            lodf_list = list(range(lodf_batch))
            lodf_batch_num += 1

    if counter > 0:
        lodf_list = lodf_list[:counter]
        lodf_concatenated = torch.cat(lodf_list, dim=1).to(torch.double).requires_grad_()
        indices1 = lodf_concatenated._indices()[0].to(torch.float32)
        indices2 = lodf_concatenated._indices()[1].to(torch.float32)
        values = lodf_concatenated._values().to(torch.float32) 

        LODF.append((indices1, indices2, values))

    end_lodf = time.time()

    time_lodf = end_lodf - start_lodf
    print('time to build lodf matrix:', time_lodf)
    

#============ pickle
import pickle

# Specify the file path in Google Drive where the dictionary will be saved
if case == 'N-1':
    file_path_lodf = 'LODF_list/IEEE118/LODF_listN1spyros.pkl'
    file_path_SL = 'LODF_list/IEEE118/SL_N1.pkl'
if case == 'N-2':
    file_path_lodf = 'LODF_list/IEEE118/LODF_listN2spyros.pkl'
    file_path_SL = 'LODF_list/IEEE118/SL_N2.pkl'
if case == 'N-3':
    file_path_lodf = 'LODF_list/IEEE118/LODF_listN3spyros.pkl'
    file_path_SL = 'LODF_list/IEEE27/SL_N3.pkl'
if case == 'N-4':
    file_path_lodf = 'LODF_list/IEEE27/LODF_listN4.pkl'
    file_path_SL = 'LODF_list/IEEE27/SL_N4.pkl'
if case == 'N-5':
    file_path_lodf = 'LODF_list/IEEE27/LODF_listN5.pkl'
    file_path_SL = 'LODF_list/IEEE27/SL_N5.pkl'

# Save the dictionary to the specified file path using pickle
with open(file_path_lodf, 'wb') as f:
    pickle.dump(LODF, f)
    

	
	
