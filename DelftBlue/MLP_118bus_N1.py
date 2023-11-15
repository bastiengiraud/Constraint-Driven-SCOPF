import sys
sys.path.append(r'/home/bgiraud')

import cvxpy as cp
import pandas as pd
import numpy as np
import math
import cvxpy_dcopf as cd
import cvxpy_scopf as cs
#import cvxpy_imb as ci
from cvxpylayers.torch import CvxpyLayer
import torch
import torchvision
import timeit
import precontingency as pc
import torch.nn as nn
import time

from tqdm import tqdm
import itertools
import torch.nn.functional as F
from torch import Tensor

from torch.utils.data import DataLoader
import torch.utils.data as Data
from sklearn.model_selection import train_test_split

import psutil
import pickle

import loadsampling as ls

#SOLVER SETTINGS FOR SCS
# used as solver for cvxpylayer in NN
# SCS is standard solver for CVXPYlayer, you can specify other solvers

SCS_solver_args={'use_indirect': False,
         'gpu': False,
         'verbose': False, # False
         'normalize': True, #True heuristic data rescaling
         'max_iters': 10000, #2500 giving the maximum number of iterations
         'scale': 100, #1 if normalized, rescales by this factor
         'eps':1e-3, #1e-3 convergence tolerance
         'cg_rate': 2, #2 for indirect, tolerance goes down like 1/iter^cg_rate
         'alpha': 1.5, #1.5 relaxation parameter
         'rho_x':1e-3, #1e-3 x equality constraint scaling
         'acceleration_lookback': 10, #10
         'write_data_filename':None}


SCS_solver_args2={'use_indirect': False,
         'gpu': False,
         'verbose': False, # False
         'normalize': True, #True heuristic data rescaling
         'max_iters': 20000, #2500 giving the maximum number of iterations
         'scale': 1, #1 if normalized, rescales by this factor
         'eps':1e-3, #1e-3 convergence tolerance
         'cg_rate': 2, #2 for indirect, tolerance goes down like 1/iter^cg_rate
         'alpha': 1.5, #1.5 relaxation parameter
         'rho_x':1e-3, #1e-3 x equality constraint scaling
         'acceleration_lookback': 10, #10
         'write_data_filename':None}


SCS_solver_args3={#'use_indirect': False,
         #'gpu': False,
         'verbose': False, # False
         'normalize': False, #True heuristic data rescaling
         'max_iters': 10000, #2500 giving the maximum number of iterations
         'scale': 1, #1 if normalized, rescales by this factor
         'eps':1e-3, #1e-3 convergence tolerance ????????
         #'cg_rate': 2, #2 for indirect, tolerance goes down like 1/iter^cg_rate ????????
         'alpha': 1.5, #1.5 relaxation parameter
         'rho_x':1e-1, #1e-3 x equality constraint scaling, there is no discrepencay in magnitude between eq and ineq right?
         'acceleration_lookback': 10, #10
         'write_data_filename':None}

ECOS_solver_args = {"solve_method":"ECOS",
    'verbose': False, # False
    'max_iters': 50000, # Maximum number of iterations
    'reltol': 1e-3, # Convergence tolerance
    # Add other ECOS solver arguments as needed
}


data = pd.read_excel('IEEE118.xlsx',sheet_name=None)
Sbase = data['par']['base'][0]
Nloads = len(data['load'])
load_loc = torch.from_numpy(np.array(data['load']['bus']))
Ngens = len(data['gen'])
gen_loc = torch.from_numpy(np.array(data['gen']['bus']))
Nlines = len(data['line'])
Nbus = len(data['bus'])

Nsamples = 1700
Niterations = 100
batch_size = 100
train = 1500/1700
tollerance_crit = 0.1
learning_rate = 0.001
wd = 0.1 
case = 'N-1'
probabilistic = 'False'
#==== cel 2

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

#====== cel 3

#data
LB = 0.75*np.ones(Nloads)
UB = 1.25*np.ones(Nloads)
X = np.array(data['load']['p'])/Sbase*ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, LB, UB, Nsamples).T

#file_path = 'trained models/training data/Xtrain118.pkl' 
#with open(file_path, 'wb') as f:
#    pickle.dump(X, f)

file_path = 'X_fixed.pkl'  
with open(file_path, 'rb') as f:
    X = pickle.load(f)

Xtrain = torch.tensor(X[0:int(Nsamples),:],dtype = torch.float32)        
Xtrain_transpose = Xtrain.transpose(0,1)
#Xtest = torch.tensor(X[int(train*Nsamples):,:],dtype = torch.float64)
#standardise
Xmin, Xmax,Xmean,Xstd = np.min(X, axis = 0),np.max(X, axis = 0),np.mean(X, axis = 0),np.std(X, axis = 0)
Xscal = torch.tensor(( X - Xmean ) / Xstd, dtype=torch.float32)
#Xscaltrain = Xscal[0:int(train*Nsamples),:]
#Xscaltest = Xscal[int(train*Nsamples):,:]

c_max = np.max(np.array(data['gen']['cost']))
c_min = np.min(np.array(data['gen']['cost']))

gencost = torch.tensor(np.array(data['gen']['cost']),dtype = torch.float32)  
c_delta = c_max - c_min
cgi0 = torch.tensor((np.array(data['gen']['cost'])-c_min)/c_delta,dtype = torch.float32)
alpha = 1

#========= cel 4

start = timeit.default_timer()
#problem0, Pd0, cg0, cll0, Pgi, Fl0, Th0, lam10, lam20 = cd.create_dcopf_correction(data)
problem0, Pd0, cll0, Pgi, Fl0, Th0, lam10, lam20 = cd.create_dcopf_correction(data)
assert problem0.is_dpp()

#cvxpylayer0 = CvxpyLayer(problem0, parameters=[Pd0, cg0, cll0], variables=[Pgi, Fl0, Th0, lam10, lam20])
cvxpylayer0 = CvxpyLayer(problem0, parameters=[Pd0, cll0], variables=[Pgi, Fl0, Th0, lam10, lam20])
time1 = timeit.default_timer() - start
#print("Time to create model: ", time1)

max_f = data["line"]['max_f'][0]/Sbase
l = len(data['line'])
#print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


#========= cel 5

import pickle
dict_method = 'pickle'

start_load = time.time()
if case == 'N-1':
    if dict_method == 'pickle': 
        file_path_lodf = 'LODF_listN1.pkl'  
        with open(file_path_lodf, 'rb') as f:
            LODF_dict = pickle.load(f)
            
        #file_path_SL = 'LODF_list/IEEE27/SL_N1.pkl'
        #with open(file_path_SL, 'rb') as f:
        #    singular_list = pickle.load(f)
    
    Ncontingencies = int(l) 
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = Ncontingencies
    last_batch_size = Ncontingencies - (lodf_batch_size*(tot_lodf_batches-1))

if case == 'N-2':
    if dict_method == 'pickle': 
        file_path = 'LODF_list/IEEE27/LODF_listN2.pkl'  
        with open(file_path, 'rb') as f:
            LODF_dict = pickle.load(f)
        
        file_path_SL = 'LODF_list/IEEE27/SL_N2.pkl'
        with open(file_path_SL, 'rb') as f:
            singular_list = pickle.load(f)

    Ncontingencies = int(l*(l-1)/2)
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = Ncontingencies
    last_batch_size = Ncontingencies - (lodf_batch_size*(tot_lodf_batches-1))
            

if case == 'N-3':        
    if dict_method == 'pickle': 
        file_path = 'LODF_list/IEEE27/LODF_listN3.pkl'  
        with open(file_path, 'rb') as f:
            LODF_dict = pickle.load(f)
        
        file_path_SL = 'LODF_list/IEEE27/SL_N3.pkl'
        with open(file_path_SL, 'rb') as f:
            singular_list = pickle.load(f)
    
    Ncontingencies = int(l*(l-1)/2*(l-2)/3)
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = Ncontingencies #20000
    last_batch_size = Ncontingencies - (lodf_batch_size*(tot_lodf_batches-1))
    
if case == 'N-4':
        
    if dict_method == 'pickle': 
        file_path = 'LODF_list/IEEE27/LODF_listN4.pkl'  
        with open(file_path, 'rb') as f:
            LODF_dict = pickle.load(f)
        
        file_path_SL = 'LODF_list/IEEE27/SL_N4.pkl'
        with open(file_path_SL, 'rb') as f:
            singular_list = pickle.load(f)
    
    Ncontingencies = int(l*(l-1)/2*(l-2)/3*(l-3)/4)
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = Ncontingencies #20000
    last_batch_size = Ncontingencies - (lodf_batch_size*(tot_lodf_batches-1))
    
if case == 'N-5':
        
    if dict_method == 'pickle': 
        file_path = 'LODF_list/IEEE27/LODF_listN5.pkl'  
        with open(file_path, 'rb') as f:
            LODF_dict = pickle.load(f)
        
        file_path_SL = 'LODF_list/IEEE27/SL_N5.pkl'
        with open(file_path_SL, 'rb') as f:
            singular_list = pickle.load(f)
    
    Ncontingencies = int(l*(l-1)/2*(l-2)/3*(l-3)/4*(l-4)/5)
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = 20000 # Ncontingencies #20000
    last_batch_size = Ncontingencies - (lodf_batch_size*(tot_lodf_batches-1))
        
    
end_load = time.time()

#print('Load LODF time:', end_load - start_load)
#print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


#=========== cel 6

#Benchmark either DCOPF or SCOPF
benchmark = 'DCOPF' #'SCOPF'

Pgisscopf = torch.zeros(Ngens,int(Nsamples),dtype = torch.float32)
Fl0sscopf = torch.zeros(Nlines,int(Nsamples),dtype = torch.float32)
Flcs = torch.zeros(Nlines,l,int(Nsamples),dtype = torch.float32)
lam1s = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)
lam2s = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)
lam1sc = torch.zeros(Nbus,l,int(Nsamples),dtype = torch.float32)
lam2sc = torch.zeros(Nbus,l,int(Nsamples),dtype = torch.float32)
Th0 = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)

#======== remove islanding cases from contingency list
lcontingencies = range(l)
islanding_cases = [15,20]
lcontingencies_filtered = [l for l in lcontingencies if l not in islanding_cases]
lcontingencies = lcontingencies_filtered

time_scopfs = time.time()

#======== determine infeasible cases
# add loads
#load_profile = torch.zeros((Nsamples,Nbus)).to(torch.float32)
#load_profile[:, load_loc] = torch.from_numpy(X).to(torch.float32)

if benchmark == 'SCOPF':
    problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf(data, lcontingencies)

elif benchmark == 'SCOPF2':
    problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc = cs.create_scopf2(data, lcontingencies)

elif benchmark == 'DCOPF': 
    #problem0scopf, Pd0scopf, cg0scopf, cl0scopf, Pgiscopf, Fl0scopf, Th0scopf, lam1dc, lam2dc = cd.create_dcopf_upper(data)
    problem0scopf, Pd0scopf, cg0scopf, cll0scopf, Pgiscopf, Fl0scopf, Th0scopf, lam1, lam2 = cd.create_dcopf_lower(data)
    cll0scopf.value = np.zeros(Ngens)

cg0scopf.value = np.array(data['gen']['cost'])
for entry in tqdm(range(int(Nsamples)),position=0, leave=True):   
    Pd0scopf.value = X_scopf[entry,:] #np.array(data['load']['p'])/Sbase    
    try:
        solution  = problem0scopf.solve(solver=cp.ECOS)    # cp.MOSEK
        if problem0scopf.status in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("Problem infeasible or unbounded")
            #print("Optimal value: %s" % problem0scopf.value)
        Pgisscopf[:,entry] = torch.tensor(Pgiscopf.value)
        Fl0sscopf[:,entry] = torch.tensor(Fl0scopf.value)
        Th0[:,entry] = torch.tensor(Th0scopf.value)

        lam1s[:,entry] = torch.tensor(lam1.value)
        lam2s[:,entry] = torch.tensor(lam2.value)
    except cp.error.SolverError:
        Pgisscopf[:,entry] = Pgisscopf[:,(entry-1)]
        Fl0sscopf[:,entry] = Fl0sscopf[:,(entry-1)]
        Th0[:,entry] = Th0[:,(entry-1)]

        lam1s[:,entry] = lam1s[:,(entry-1)]
        lam2s[:,entry] = lam2s[:,(entry-1)]

        Xtrain[entry,:] = Xtrain[(entry-1),:]

        print("Solver unable to solve for entry:", entry)
        continue
    
time_scopfs = time.time() - time_scopfs
cost_scopf = torch.matmul(gencost,Pgisscopf)

#==========================

with torch.set_grad_enabled(True):
    N, D_in, D_out = Nsamples, Nloads, Ngens
    H1, H2, H3 = 8,8,8
   
    #very large network
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H1),
        torch.nn.ReLU(),
        nn.Dropout(p=0.2),
        torch.nn.Linear(H1, H2),
        torch.nn.ReLU(),
        nn.Dropout(p=0.2),
        torch.nn.Linear(H2, H3),
        torch.nn.ReLU(),
        nn.Dropout(p=0.2),
        torch.nn.Linear(H3, D_out), #before 23/12 when everything was working!
        #torch.nn.Sigmoid(),
        torch.nn.Tanh(),
        #torch.nn.Softmax(dim=0),
    )
    
    #model = torch.nn.RNN(input_size = D_in, hidden_size = D_out, num_layers=3, nonlinearity = 'tanh', batch_first = True)
    model.double()
    
    def init_weights(u):
        if type(u) == nn.Linear:
            torch.nn.init.xavier_uniform_(u.weight)
            #u.reset_parameters() # default gaussion initializaiton
            torch.nn.init.xavier_uniform_(u.weight, gain=nn.init.calculate_gain('relu'))
            u.bias.data.fill_(0)    
    model.apply(init_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #optimizer = DoG(model.parameters())
    
    model.to(torch.float32)
    print(model)

print('Load LODF time:', end_load - start_load)
#print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    
num_params_full = sum(p.numel() for p in model.parameters())
#num_params_TT = sum(p.numel() for p in modelTT.parameters())
print('')
print('FC neural network:', num_params_full)
#print('TT neural network:', num_params_TT)
print('input loads:', D_in, 'output gens:', D_out)



#======================

if device_ids:
    model = DataParallel(model, device_ids=device_ids)
    model = model.to(device)
    print(device_ids)

model.to(torch.float32)
print(next(model.parameters()).device)
print(next(model.parameters()).dtype)

# prompt to empty GPU cache: torch.cuda.empty_cache()

start = time.time()
max_p = torch.from_numpy(np.array(-data["gen"]['min_p']/Sbase))
min_p = torch.from_numpy(np.array(data["gen"]['max_p']/Sbase))

#============== Initialize training testing data
Xscal_train, Xscal_test, cost_scopf_train, cost_scopf_test = train_test_split(Xscal, cost_scopf, test_size=(200/1700), random_state=42)

training_dataset = Data.TensorDataset(Xscal_train, cost_scopf_train)
testing_dataset = Data.TensorDataset(Xscal_test, cost_scopf_test)

train_loader = Data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)  # Note: shuffle=False for testing

import torch.nn.functional as F

flow_penalty = 100
crit_penalty = 0
infeasibility_penalty = 0
cost_penalty = 1

#================ Initialize tensors
error = np.zeros(shape=(Niterations,2))
avg_infeasibility = np.zeros(shape=(Niterations,2))
relcost = np.zeros(shape=(Niterations,2))

time_CF = np.zeros(shape=(Niterations,1))
memory_CF = np.zeros(shape=(Niterations,1))
percentage_CF = np.zeros(shape=(Niterations,1))
backward_CF = np.zeros(shape=(Niterations,1))

Pgis = torch.zeros(size = (Ngens,int(batch_size)),dtype = torch.float32)
Flis = torch.zeros(size = (Nlines,int(batch_size)),dtype = torch.float32)
clil = torch.zeros(size = (Ngens,int(batch_size)),dtype = torch.float32)
infeasibility = torch.zeros(int(batch_size), requires_grad = False)

relcotrain_epoch = torch.zeros(size = (int(Nsamples*(1500/1700)/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)
relcotest_epoch = torch.zeros(size = (int(Nsamples*(200/1700)/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)

imbalancetrain_epoch = torch.zeros(size = (int(Nsamples*(1500/1700)/batch_size),1),dtype = torch.float32, requires_grad = False)
imbalancetest_epoch = torch.zeros(size = (int(Nsamples*(200/1700)/batch_size),1),dtype = torch.float32, requires_grad = False)

infeasibilitytrain_epoch = torch.zeros(size = (int(Nsamples*(1500/1700)/batch_size),Nbus),dtype = torch.float32, requires_grad = False)
infeasibilitytest_epoch = torch.zeros(size = (int(Nsamples*(200/1700)/batch_size),Nbus),dtype = torch.float32, requires_grad = False)
infeasibility_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)

line_violation_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
crit_violation_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
avg_violation_list = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
count_violation_list = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)

violation_percentage = torch.zeros(size = (int(Nsamples*train/batch_size),1),dtype = torch.float32, requires_grad = False)
time_contflow = torch.zeros(size = (int(Nsamples*train/batch_size),1),dtype = torch.float32, requires_grad = False)
memory_contflow = torch.zeros(size = (int(Nsamples*train/batch_size),1),dtype = torch.float32, requires_grad = False)
backward_contflow = torch.zeros(size = (int(Nsamples*train/batch_size),1),dtype = torch.float32, requires_grad = False)


print("It |   Time   |    Imbalance    | Mod out  |   Infeasibility   |  Rel cost")
for epoch in range(Niterations):
    
    time_nnSCS = time.time()
    for i, (Xbatch, cost_scopf_batch) in enumerate(tqdm(train_loader)):
            start_forward = time.time()
            start_forward_memory = psutil.virtual_memory()[3] / 1000000000            
            
            #============= initialize
            model.train()
            optimizer.zero_grad() 
            model.zero_grad()
            Xbatch = Xbatch.clone().detach().to(device)
            cost_scopf_batch = cost_scopf_batch.clone().detach().to(device)

            Xtrain_batch = ((Xbatch.cpu() * Xstd) + Xmean).to(torch.float32).to(device)
            load_profile = torch.zeros((batch_size, Nbus)).to(device)
            #load_profile[:, load_loc] = Xtrain_batch

            #================== Perform forward pass
            start_FP = time.time()
            clil = (model(Xbatch) + 1) / 2
            #clil = (model(Xbatch))
            end_FP = time.time()
            
            start_cvxpy = time.time()
            #Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtrain_batch, gencost, clil, solver_args=ECOS_solver_args)
            Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtrain_batch, clil, solver_args=SCS_solver_args3)
            Pgis = Pgi.T
            Flis = Fli.T.to(torch.float32)
            end_cvxpy = time.time()
            
            #=============== modify load profile with generator outputs
            #load_profile[:, gen_loc] += Pgi
            #nonzero_values, _ = torch.min(load_profile.masked_fill(load_profile == 0, float('inf')), dim=1, keepdim=True)
            #mask = load_profile == 0
            #load_profile = torch.where(mask, nonzero_values, load_profile)

            #================ infeasibility
            infeasibility_tensor = (lam1i + lam2i)
            infeasibility_loss = infeasibility_tensor.sum() / int(batch_size)
            infeasibility_batch = infeasibility_tensor.sum(dim=0).detach()

            infeasibility_tensor[abs(infeasibility_tensor) < 1e-5] = 0
            infeasible_buses = (abs(infeasibility_tensor) > 1e-5)

            infeasible_cases = torch.zeros((batch_size, 1)).to(device)
            row_has_true = torch.any(infeasible_buses, axis=1)
            infeasible_cases[row_has_true] = True
            infeasible_cases = infeasible_cases.sum()

            #==================== cost
            gencost = gencost.requires_grad_()
            
            batch_cost = torch.matmul(gencost, Pgis)
            batch_relco = ((batch_cost - cost_scopf_batch) / cost_scopf_batch) * 100
            relco_avg = torch.mean(batch_relco)

            cost_loss = torch.mean(batch_cost) #/ int(batch_size)

            #===================== compute contingency flows for each sample  
            imbalancetrain_batch = 0
            critical_loss = 0
            count_crit_violation = 0
            count_line_violation = 0
            count_violation = 0

            for b in range(tot_lodf_batches):
                lodf_batch = lodf_batch_size
                if b == (tot_lodf_batches - 1): 
                    lodf_batch = last_batch_size
                left_ones = torch.ones((lodf_batch, 1)).to(device)
                right_ones = torch.ones((int(l * batch_size), 1)).to(device)
                Fli_0 = Flis.expand(lodf_batch, -1, -1)

                index1 = LODF_dict[b][0].detach()
                index2 = LODF_dict[b][1].detach()
                indices = torch.stack((index1, index2)).detach()
                values = LODF_dict[b][2].detach()
                shape = (l, int(l * lodf_batch))
                lodf = torch.sparse_coo_tensor(indices, values, size=shape, requires_grad=False).T.to(device).detach()

                imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis).reshape((lodf_batch, l, batch_size))) - max_f)).detach().to(device)
                #imbalancetrain_batch_loop_real = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis).reshape((lodf_batch, l, batch_size))) - max_f)).detach().to(device)
    
                count_crit_violation_loop = torch.sum(imbalancetrain_batch_loop > tollerance_crit * max_f, dim=1)
                count_line_violation_loop = torch.sum(imbalancetrain_batch_loop > 0.01*max_f, dim=1)
                imbalancetrain_batch_loop = imbalancetrain_batch_loop.reshape(int(l * lodf_batch), -1)

                mask = torch.all(imbalancetrain_batch_loop == 0, dim=1) 
                indices_remove = mask.nonzero().squeeze(1)
                num_zero_rows = torch.sum(mask)
                batch_percentage_violation = (num_zero_rows.item() / len(imbalancetrain_batch_loop[:, 0])) * 100

                mask_flow = torch.ones(imbalancetrain_batch_loop.shape[0], dtype=torch.bool).to(device)
                mask_flow[indices_remove] = False
                lodf_test = lodf.to_dense()[mask_flow]
                lodf_test = lodf_test.clone().detach()
                lodf = lodf_test.to_sparse_coo().requires_grad_().to(device)


                Fli_0 = Flis.repeat(lodf_batch, 1).to(torch.float32).to(device)[mask_flow].to(device)
                imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis)) - max_f)).to(device)
                critical_violation_penalty = imbalancetrain_batch_loop[imbalancetrain_batch_loop > tollerance_crit * max_f]
                count_violation_loop = torch.count_nonzero(imbalancetrain_batch_loop)  

                imbalancetrain_batch += torch.sum(imbalancetrain_batch_loop) / int(batch_size)
                #imbalancetrain_batch += (torch.norm(imbalancetrain_batch_loop, dim = 0, p = 1)).mean()
                critical_loss += torch.sum(critical_violation_penalty)/int(batch_size)
                
                count_violation += count_violation_loop
                count_crit_violation += torch.count_nonzero(count_crit_violation_loop.detach())
                count_line_violation += torch.count_nonzero(count_line_violation_loop.detach())

            end_forward = time.time()
            end_forward_memory = psutil.virtual_memory()[3] / 1000000000
            
            memory_contflow_batch = end_forward_memory - start_forward_memory
            time_contflow_batch = end_forward - start_forward                              
            
            #======================== Perform back pass
            loss = flow_penalty * imbalancetrain_batch + infeasibility_penalty * infeasibility_loss + cost_penalty * cost_loss 
            
            start_BP = time.time()
            loss.backward()
            end_BP = time.time()
            backward_batch = end_BP - start_BP
            #print('Backward pass:', end_BP - start_BP)
            
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            relcotrain_epoch[i,:] = batch_relco.detach()
            imbalancetrain_epoch[i,:] = imbalancetrain_batch.detach()
            infeasibilitytrain_epoch[i,:] = infeasibility_batch.detach()
            infeasibility_count[i,:] = infeasible_cases.detach()
            line_violation_count[i,:] = count_line_violation.detach()
            crit_violation_count[i,:] = count_crit_violation.detach()
            #avg_violation_list[i,:] = avg_violation.detach()
            count_violation_list[i,:] = count_violation
            
            violation_percentage[i,:] = batch_percentage_violation
            time_contflow[i,:] = time_contflow_batch
            memory_contflow[i,:] = memory_contflow_batch
            backward_contflow[i,:] = backward_batch
    
    with torch.no_grad():
        for i, (Xbatch, cost_scopf_batch) in enumerate(tqdm(test_loader)):
                
                model.eval() # deactivate dropout

                Xbatch = Xbatch.clone().detach().to(device)
                cost_scopf_batch = cost_scopf_batch.clone().detach().to(device)

                Xtest_batch = ((Xbatch.cpu() * Xstd) + Xmean).to(torch.float32).to(device)

                #================== Perform forward pass
                clil = (model(Xbatch) + 1) / 2
                #clil = (model(Xbatch))
                
                #Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, gencost, clil, solver_args=ECOS_solver_args)
                Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, clil, solver_args=SCS_solver_args3)
                Pgis = Pgi.T
                Flis = Fli.T.to(torch.float32)

                #================ compute cost and infeasibility
                infeasibility_batch = torch.sum(lam1i) + torch.sum(lam2i)

                batch_cost = torch.matmul(gencost, Pgis)
                batch_relco = ((batch_cost - cost_scopf_batch) / cost_scopf_batch) * 100

                #===================== compute contingency flows for each sample
                imbalancetest_batch = 0

                for b in range(tot_lodf_batches):
                    lodf_batch = lodf_batch_size
                    if b == (tot_lodf_batches - 1): 
                        lodf_batch = last_batch_size
                    left_ones = torch.ones((lodf_batch, 1)).to(device)
                    right_ones = torch.ones((int(l * batch_size), 1)).to(device)
                    Fli_0 = Flis.expand(lodf_batch, -1, -1).detach()

                    index1 = LODF_dict[b][0].detach()
                    index2 = LODF_dict[b][1].detach()
                    indices = torch.stack((index1, index2)).detach()
                    values = LODF_dict[b][2].detach()
                    shape = (l, int(l * lodf_batch))
                    lodf = torch.sparse_coo_tensor(indices, values, size=shape).T.to(device)

                    imbalancetest_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis).reshape((lodf_batch, l, batch_size))) - max_f))
                    imbalancetest_batch += torch.sum(imbalancetest_batch_loop) / int(batch_size)

        
                relcotest_epoch[i,:] = batch_relco.detach()
                imbalancetest_epoch[i,:] = imbalancetest_batch.detach()
                infeasibilitytest_epoch[i,:] = infeasibility_batch.detach()
            
    time_nnSCS = time.time() - time_nnSCS  

    imbalancetrain = torch.mean(imbalancetrain_epoch).detach()
    imbalancetest = torch.mean(imbalancetest_epoch).detach()
    
    infeasibilitytrain = torch.mean(infeasibilitytrain_epoch).detach()
    infeasibilitytest = torch.mean(infeasibilitytest_epoch).detach()
    
    relcotrain = torch.mean(relcotrain_epoch).detach()
    relcotest = torch.mean(relcotest_epoch).detach()
    
    #====== convergence criteria
    infeasible_cases_tot = torch.sum(infeasibility_count)
    line_violation_tot = torch.sum(line_violation_count)
    crit_violation_tot = torch.sum(crit_violation_count)
    #tot_avg_violation = torch.sum(avg_violation_list).detach()
    count_violation_tot = torch.sum(count_violation_list).detach()
    
    tot_violation_percentage = torch.mean(violation_percentage)
    tot_time_contflow = torch.sum(time_contflow)
    tot_memory_contflow = torch.mean(memory_contflow)
    tot_backward_contflow = torch.sum(backward_contflow)
    
    #============= 
    error[epoch,:] = torch.stack((imbalancetrain,imbalancetest)).detach().numpy()
    avg_infeasibility[epoch,:] =  torch.stack((infeasibilitytrain,infeasibilitytest)).detach().numpy()
    relcost[epoch,:] =  torch.stack((relcotrain,relcotest)).detach().numpy()
    
    #================= computational graph
    time_CF[epoch,:] = tot_time_contflow
    memory_CF[epoch,:] = tot_memory_contflow 
    percentage_CF[epoch,:] = tot_violation_percentage
    backward_CF[epoch,:] = tot_backward_contflow

    print(epoch, " | ", round(time_nnSCS,0), " | ",np.round(error[epoch,:],3), " | ", np.round(torch.mean(clil[:]).cpu().detach().numpy(),3), " | ", np.round(avg_infeasibility[epoch,:],3), " | ", np.round(relcost[epoch,:], 3) )
    #scheduler.step()
    #print(f"Learning Rate: {scheduler.get_lr()[0]}")
    
    #======================= Convergence check infeasible cases
    print('current infeasibility:', (infeasible_cases_tot/int(train*Nsamples))*100, '%', 'goal:', 1, '%')
    
    print('infeasible contingency cases:', line_violation_tot, (line_violation_tot/(Ncontingencies*Nsamples))*100, '%')
    print('10%+ contingency cases:', crit_violation_tot, (crit_violation_tot/(Ncontingencies*Nsamples))*100, '%')
    print('number of violations:', count_violation_tot)
    print('percentage lines not in violation:', tot_violation_percentage, '%')
    
    print('flow penalty:', flow_penalty*imbalancetrain_batch)
    print('infeasible penalty:', infeasibility_penalty*infeasibility_loss)
    print('cost penalty:', cost_penalty*cost_loss)
    print('')
    
    
    
        
end = time.time()
print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)











