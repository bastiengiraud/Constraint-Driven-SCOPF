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


data = pd.read_excel('IEEE118spyros.xlsx',sheet_name=None)
Sbase = data['par']['base'][0]
Nloads = len(data['load'])
load_loc = torch.from_numpy(np.array(data['load']['bus']))
Ngens = len(data['gen'])
gen_loc = torch.from_numpy(np.array(data['gen']['bus']))
Nlines = len(data['line'])
Nbus = len(data['bus'])

Nsamples = 3200
Niterations = 100
batch_size = 50
train = 3000/3200
test = 200/3200
tollerance_crit = 0.1
learning_rate = 0.001
wd = 0.1 
case = 'N-3'
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
#LB = 0.75*np.ones(Nloads)
#UB = 1.25*np.ones(Nloads)
#X = np.array(data['load']['p'])/Sbase*ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, LB, UB, Nsamples).T

#file_path = 'trained models/training data/Xtrain118.pkl' 
#with open(file_path, 'wb') as f:
#    pickle.dump(X, f)

file_path = 'Xtrain_118_spyros.pkl'  
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

PTDF = torch.from_numpy(pc.compptdfs(data)).to(torch.float32)

#========= cel 4

start = timeit.default_timer()
#problem0, Pd0, cg0, cll0, Pgi, Fl0, Th0, lam10, lam20 = cd.create_dcopf_correction(data)
problem0, Pd0, cll0, Pgi, Fl0, Th0, lam10, lam20 = cd.create_dcopf_correction(data)
assert problem0.is_dpp()

#cvxpylayer0 = CvxpyLayer(problem0, parameters=[Pd0, cg0, cll0], variables=[Pgi, Fl0, Th0, lam10, lam20])
cvxpylayer0 = CvxpyLayer(problem0, parameters=[Pd0, cll0], variables=[Pgi, Fl0, Th0, lam10, lam20])
time1 = timeit.default_timer() - start
#print("Time to create model: ", time1)

max_f = torch.from_numpy(np.array(data["line"]['max_f']/Sbase)).unsqueeze(1).repeat(1,batch_size)
l = len(data['line'])
#print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)


#========= cel 5

import pickle
dict_method = 'pickle'

start_load = time.time()
if case == 'N-1':
    if dict_method == 'pickle': 
        file_path_lodf = 'LODF_listN1spyros.pkl'  
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
        file_path = 'LODF_listN2spyros.pkl'  
        with open(file_path, 'rb') as f:
            LODF_dict = pickle.load(f)
        
        #file_path_SL = 'LODF_list/IEEE27/SL_N2.pkl'
        #with open(file_path_SL, 'rb') as f:
        #    singular_list = pickle.load(f)

    Ncontingencies = int(l*(l-1)/2)
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = Ncontingencies
    last_batch_size = Ncontingencies - (lodf_batch_size*(tot_lodf_batches-1))
            

if case == 'N-3':        
    if dict_method == 'pickle': 
        file_path = 'LODF_listN3spyros.pkl'  
        with open(file_path, 'rb') as f:
            LODF_dict = pickle.load(f)
        
        #file_path_SL = 'LODF_list/IEEE27/SL_N3.pkl'
        #with open(file_path_SL, 'rb') as f:
        #    singular_list = pickle.load(f)
    
    Ncontingencies = int(l*(l-1)/2*(l-2)/3)
    tot_lodf_batches = len(LODF_dict)
    lodf_batch_size = 20000
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
    Pd0scopf.value = X[entry,:] #np.array(data['load']['p'])/Sbase    
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
    H1, H2, H3 = 16,16,16
   
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

gen_indices = data['gen']['bus'] 
generator_nodes = np.zeros(Nbus)
for index in gen_indices:
    generator_nodes[index] = 1
gen_nodes = np.nonzero(generator_nodes)
gen_nodes = gen_nodes[0].tolist()
print(gen_nodes)

load_indices = data['load']['bus'] 
load_nodes = np.zeros(Nbus)
for index in load_indices:
    load_nodes[index] = 1
load_nodes = np.nonzero(load_nodes)
load_nodes = load_nodes[0].tolist()
print(load_nodes)

#=====================

# prompt to empty GPU cache: torch.cuda.empty_cache()

start = time.time()
max_p = torch.from_numpy(np.array(-data["gen"]['min_p']/Sbase))
min_p = torch.from_numpy(np.array(data["gen"]['max_p']/Sbase))

#============== Initialize training testing data
Xscal_train, Xscal_test, cost_scopf_train, cost_scopf_test = train_test_split(Xscal, cost_scopf, test_size=(test), random_state=42)

training_dataset = Data.TensorDataset(Xscal_train, cost_scopf_train)
testing_dataset = Data.TensorDataset(Xscal_test, cost_scopf_test)

train_loader = Data.DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
test_loader = Data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)  # Note: shuffle=False for testing

import torch.nn.functional as F

flow_penalty = 1
crit_penalty = 0
infeasibility_penalty = 0
cost_penalty = 1
soft_penalty = 100

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

relcotrain_epoch = torch.zeros(size = (int(Nsamples*(train)/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)
relcotest_epoch = torch.zeros(size = (int(Nsamples*(test)/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)

imbalancetrain_epoch = torch.zeros(size = (int(Nsamples*(train)/batch_size),1),dtype = torch.float32, requires_grad = False)
imbalancetest_epoch = torch.zeros(size = (int(Nsamples*(test)/batch_size),1),dtype = torch.float32, requires_grad = False)

infeasibilitytrain_epoch = torch.zeros(size = (int(Nsamples*(train)/batch_size),Nbus),dtype = torch.float32, requires_grad = False)
infeasibilitytest_epoch = torch.zeros(size = (int(Nsamples*(test)/batch_size),Nbus),dtype = torch.float32, requires_grad = False)
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
            start_forward_memory = 0#psutil.virtual_memory()[3] / 1000000000            
            
            #============= initialize
            model.train()
            optimizer.zero_grad() 
            model.zero_grad()
            Xbatch = Xbatch.clone().detach().to(device)
            cost_scopf_batch = cost_scopf_batch.clone().detach().to(device)

            Xtrain_batch = ((Xbatch.cpu() * Xstd) + Xmean).to(torch.float32).to(device)

            #================== Perform forward pass
            start_FP = time.time()
            clil = (model(Xbatch) + 1) / 2
            #clil = (model(Xbatch))
            end_FP = time.time()
            
            #============= soft violation base case
            Pg_guess = torch.mul(max_p,clil) 
            
            Pg_train = torch.zeros((batch_size,Nbus)).to(torch.float32)
            Pd_train = torch.zeros((batch_size,Nbus)).to(torch.float32)

            Pg_train[:,gen_nodes] = Pg_guess.to(torch.float32)
            Pd_train[:,load_nodes] = Xtrain_batch.to(torch.float32)
                
            Fl_base = F.relu(torch.abs(PTDF@(Pg_train-Pd_train).T)-max_f).sum()/int(batch_size)
            
            #======================== cvxpylayer
            start_cvxpy = time.time()
            #Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtrain_batch, gencost, clil, solver_args=ECOS_solver_args)
            Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtrain_batch, clil, solver_args=SCS_solver_args3)
            Pgis = Pgi.T
            Flis = Fli.T.to(torch.float32)
            end_cvxpy = time.time()
        

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
                F_max = max_f.expand(lodf_batch, -1, -1)

                index1 = LODF_dict[b][0].detach()
                index2 = LODF_dict[b][1].detach()
                indices = torch.stack((index1, index2)).detach()
                values = LODF_dict[b][2].detach()
                shape = (l, int(l * lodf_batch))
                lodf = torch.sparse_coo_tensor(indices, values, size=shape, requires_grad=False).T.to(device).detach()

                imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis).reshape((lodf_batch, l, batch_size))) - F_max)).detach().to(device)
                
                count_crit_violation_loop = torch.sum(imbalancetrain_batch_loop > tollerance_crit * max_f, dim=1)
                count_line_violation_loop = torch.sum(imbalancetrain_batch_loop > 0, dim=1)
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
                F_max = max_f.repeat(lodf_batch, 1).to(torch.float32).to(device)[mask_flow].to(device)
                imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis)) - F_max)).to(device)
                critical_violation_penalty = imbalancetrain_batch_loop[imbalancetrain_batch_loop > tollerance_crit * F_max]
                count_violation_loop = torch.count_nonzero(imbalancetrain_batch_loop)  

                imbalancetrain_batch += torch.sum(imbalancetrain_batch_loop) / int(batch_size)
                critical_loss += torch.sum(critical_violation_penalty)/int(batch_size)
                
                count_violation += count_violation_loop
                count_crit_violation += torch.count_nonzero(count_crit_violation_loop.detach())
                count_line_violation += torch.count_nonzero(count_line_violation_loop.detach())
                
            end_forward = time.time()
            end_forward_memory = 0#psutil.virtual_memory()[3] / 1000000000
            
            memory_contflow_batch = end_forward_memory - start_forward_memory
            time_contflow_batch = end_forward - start_forward                              
            
            #======================== Perform back pass
            loss = flow_penalty * imbalancetrain_batch + infeasibility_penalty * infeasibility_loss + cost_penalty * cost_loss + soft_penalty*Fl_base

            start_BP = time.time()
            loss.backward()
            end_BP = time.time()
            backward_batch = end_BP - start_BP
            #print('Backward pass:', end_BP - start_BP)
            
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
                    F_max = max_f.expand(lodf_batch, -1, -1)

                    index1 = LODF_dict[b][0].detach()
                    index2 = LODF_dict[b][1].detach()
                    indices = torch.stack((index1, index2)).detach()
                    values = LODF_dict[b][2].detach()
                    shape = (l, int(l * lodf_batch))
                    lodf = torch.sparse_coo_tensor(indices, values, size=shape).T.to(device)

                    imbalancetest_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf, Flis).reshape((lodf_batch, l, batch_size))) - F_max))
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



#===============================

if case == 'N-1':
    file_path = 'Trained models/MLP_ieee118_spyros_N1_16neurons.pt'
if case == 'N-2':
    file_path = 'Trained models/MLP_ieee118_spyros_N2_16neurons.pt'
if case == 'N-3':
    file_path = 'Trained models/MLP_ieee118_spyros_N3_16neurons.pt'


# Save the model
torch.save(model.state_dict(), file_path)


print('------------------------ screening -------------------------------')
#============================
Nsamples = 1000
#X = np.array(data['load']['p'])/Sbase*ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, LB, UB, Nsamples).T

file_path = 'Xtest1_118_spyros.pkl'  
with open(file_path, 'rb') as f:
    X = pickle.load(f)
    
Xtrain = torch.tensor(X[0:int(Nsamples),:],dtype = torch.float32)        
Xtrain_transpose = Xtrain.transpose(0,1)
#Xtest = torch.tensor(X[int(train*Nsamples):,:],dtype = torch.float64)
#standardise
Xmin, Xmax,Xmean,Xstd = np.min(X, axis = 0),np.max(X, axis = 0),np.mean(X, axis = 0),np.std(X, axis = 0)
Xscal = torch.tensor(( X - Xmean ) / Xstd, dtype=torch.float32)


#===============================
#=============== Benchmark #1 contingency screening N-k
benchmark = 'SCOPF3screening' #'SCOPF'

if benchmark == 'SCOPF':
    lcontingencies = range(l)
    islanding_cases = [6, 8, 112, 132, 133, 175, 176, 182, 183]
    lcontingencies_filtered = [l for l in lcontingencies if l not in islanding_cases]
    lcontingencies = lcontingencies_filtered
    Nk_contingencies = lcontingencies
elif benchmark == 'SCOPF1screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),1)))
elif benchmark == 'SCOPF2screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),2)))
elif benchmark == 'SCOPF3screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),3)))
elif benchmark == 'SCOPF4screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),4)))
elif benchmark == 'SCOPF5screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),5)))

Pgisscopf = torch.zeros(Ngens,int(Nsamples),dtype = torch.float32)
Fl0sscopf = torch.zeros(Nlines,int(Nsamples),dtype = torch.float32)
#Flcs = torch.zeros(Nlines,len(Nk_contingencies),int(Nsamples),dtype = torch.float32)
lam1s = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)
lam2s = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)
#lam1sc = torch.zeros(Nbus,len(Nk_contingencies),int(Nsamples),dtype = torch.float32)
#lam2sc = torch.zeros(Nbus,len(Nk_contingencies),int(Nsamples),dtype = torch.float32)
Th0 = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)

#======== remove islanding cases from contingency list
if benchmark != 'SCOPF':
    lcontingencies = [0]
screening_iterations = 4

time_scopfs = time.time()

for i in range(screening_iterations):
    imbalance_list = []
    Nk_zeros = list(range(len(Nk_contingencies)))
    zero_indices = [l for l in Nk_zeros if l not in lcontingencies]
    print(lcontingencies)

    if benchmark == 'SCOPF':
        problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf(data, lcontingencies)
    elif benchmark == 'SCOPF1screening':
        problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf1_screening(data, lcontingencies)
    elif benchmark == 'SCOPF2screening':
        problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf2_screening(data, lcontingencies)
    elif benchmark == 'SCOPF3screening':
        problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf3_screening(data, lcontingencies)
    elif benchmark == 'SCOPF4screening':
        problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf4_screening(data, lcontingencies)
    elif benchmark == 'SCOPF5screening':
        problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf5_screening(data, lcontingencies)
   

    cg0scopf.value = np.array(data['gen']['cost'])
    for entry in tqdm(range(int(Nsamples)),position=0, leave=True):   
        Pd0scopf.value = X[entry,:] #np.array(data['load']['p'])/Sbase    
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
            
            if i == (screening_iterations - 1):
                Xtrain[entry,:] = Xtrain[(entry-1),:]
            
            print("Solver unable to solve for entry:", entry)
            continue
        
    
    cost_scopf = torch.matmul(gencost,Pgisscopf)
    
    if benchmark == 'SCOPF':
        break

    #============ compute Nk flows
    Flis = Fl0sscopf
                
    for b in (range(tot_lodf_batches)):

        lodf_batch = lodf_batch_size
        if b == (tot_lodf_batches - 1):
            lodf_batch = last_batch_size
        left_ones = torch.ones((lodf_batch,1))
        right_ones = torch.ones((int(l*batch_size),1))
        Fli_0 = Flis.expand(lodf_batch, -1, -1)

        summed_tensor = torch.zeros(lodf_batch).to(device)

        index1 = LODF_dict[b][0].detach()
        index2 = LODF_dict[b][1].detach()
        indices = torch.stack((index1,index2)).detach()
        values = LODF_dict[b][2].detach()
        shape = (l,int(l*lodf_batch))
        lodf = torch.sparse_coo_tensor(indices,values,size = shape, requires_grad = False).T.to(device).detach()

        for k in range(int(Nsamples/batch_size)):
            start = k*batch_size
            end = (k+1)*batch_size
            imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0[:,:,start:end] + torch.sparse.mm(lodf,Flis[:,start:end]).reshape((lodf_batch,l,batch_size)))- max_f)).detach()

            #======= get indices of contingencies of highest violations
            summed_tensor += torch.sum(torch.sum(imbalancetrain_batch_loop, dim=1), dim=1)

        if b == 0:
            desired_length = len(summed_tensor)

        padding_length = desired_length - len(summed_tensor)
        padded_tensor = F.pad(summed_tensor, (0, padding_length), value=0)
        imbalance_list.extend(padded_tensor)


    indices_per_iteration = 20 # len(summed_tensor)

    stacked_imbalance = torch.stack(imbalance_list, dim=0)
    highest_indices = (np.argsort(stacked_imbalance.cpu())[-indices_per_iteration:]).tolist()
    new_indices = [idx for idx in highest_indices if idx not in lcontingencies] # Filter out the indices that are already in the list
    lcontingencies.extend(new_indices) # Append the new indices to the list

    
time_scopfs = time.time() - time_scopfs
print('contingency screening time:', time_scopfs)

#============================

##============= check infeasibility
infeasibility_base = (lam1s + lam2s).permute(1,0)
infeasibility_base[abs(infeasibility_base) < 1e-4] = 0
infeasible_buses_base = (abs(infeasibility_base) > 1e-4) #0.001*load_profile) # 10% of total load available for generation/load shedding
scopf_cost_average = cost_scopf.mean()

infeasible_cases_base = torch.zeros((Nsamples,1))
row_has_true_base = torch.any(infeasible_buses_base, axis=1)
infeasible_cases_base[row_has_true_base] = True
infeasible_cases_tot_base = infeasible_cases_base.sum()
print('infeasible base cases:', infeasible_cases_tot_base)
print('percentage base infeasibility:', (infeasible_cases_tot_base/(Nsamples))*100, '%')
print('total average cost:', scopf_cost_average)

#=========== check how many violations N-k
Flis = Fl0sscopf
    
count_crit_violation = 0
count_line_violation = 0

for b in (range(tot_lodf_batches)):

        lodf_batch = lodf_batch_size
        if b == (tot_lodf_batches - 1):
            lodf_batch = last_batch_size
        left_ones = torch.ones((lodf_batch,1))
        right_ones = torch.ones((int(l*batch_size),1))
        Fli_0 = Flis.expand(lodf_batch, -1, -1)

        summed_tensor = torch.zeros(lodf_batch).to(device)

        index1 = LODF_dict[b][0].detach()
        index2 = LODF_dict[b][1].detach()
        indices = torch.stack((index1,index2)).detach()
        values = LODF_dict[b][2].detach()
        shape = (l,int(l*lodf_batch))
        lodf = torch.sparse_coo_tensor(indices,values,size = shape, requires_grad = False).T.to(device).detach()

        for k in range(int(Nsamples/batch_size)):
            start = k*batch_size
            end = (k+1)*batch_size
            imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0[:,:,start:end] + torch.sparse.mm(lodf,Flis[:,start:end]).reshape((lodf_batch,l,batch_size)))- max_f)).detach()

            count_crit_violation_loop = torch.sum(imbalancetrain_batch_loop > tollerance_crit*max_f, dim=1)
            count_line_violation_loop = torch.sum(imbalancetrain_batch_loop > 0, dim=1)

            count_crit_violation += torch.count_nonzero(count_crit_violation_loop)
            count_line_violation += torch.count_nonzero(count_line_violation_loop)

crit_violation_tot = count_crit_violation
line_violation_tot = count_line_violation

print('infeasible N-k contingency cases:', line_violation_tot, (line_violation_tot/(Ncontingencies*Nsamples))*100, '%')
print('10%+ contingency cases:', crit_violation_tot, (crit_violation_tot/(Ncontingencies*Nsamples))*100, '%')



#=-=========

start_test = time.time()
#============== Initialize training testing data
testing_dataset = Data.TensorDataset(Xscal, cost_scopf)
test_loader = Data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)  # Note: shuffle=False for testing

#================ Initialize tensors
error = np.zeros(shape=(Niterations,1))
avg_infeasibility = np.zeros(shape=(Niterations,1))
relcost = np.zeros(shape=(Niterations,1))
cost = np.zeros(shape=(Niterations,1))

Pgis = torch.zeros(size = (Ngens,int(batch_size)),dtype = torch.float32)
Flis = torch.zeros(size = (Nlines,int(batch_size)),dtype = torch.float32)
clil = torch.zeros(size = (Ngens,int(batch_size)),dtype = torch.float32)
infeasibility = torch.zeros(int(batch_size), requires_grad = False)

relcotest_epoch = torch.zeros(size = (int(Nsamples/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)
cost_epoch = torch.zeros(size = (int(Nsamples/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)

imbalancetest_epoch = torch.zeros(size = (round((Nsamples)/batch_size),1),dtype = torch.float32, requires_grad = False)

infeasibilitytest_epoch = torch.zeros(size = (int(Nsamples/batch_size),Nbus),dtype = torch.float32, requires_grad = False)
infeasibility_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
line_violation_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
crit_violation_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
avg_violation_list = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
count_violation_list = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)

print("It |   Time   |    Imbalance    | Mod out  |   Infeasibility   |  Rel cost")
for epoch in range(1):    
    with torch.no_grad():
        for i, (Xbatch, cost_scopf_batch) in enumerate(tqdm(test_loader)):
                
                model.eval() # deactivate dropout, when I changed this, testing error stayed same

                Xbatch = Xbatch.clone().detach()
                cost_scopf_batch = cost_scopf_batch.clone().detach()

                Xtest_batch = ((Xbatch.cpu() * Xstd)+Xmean).to(torch.float32).to(device)
                load_profile = torch.zeros((batch_size,Nbus)).to(device)
                load_profile[:, load_loc] = Xtest_batch

                #================== Perform forward pass
                #clil = (model(Xbatch))
                clil = (model(Xbatch)+1)/2
                
                #Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, gencost, clil, solver_args=ECOS_solver_args)
                try:
                    Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, clil, solver_args=ECOS_solver_args)
                except:
                    Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, clil, solver_args=SCS_solver_args3)
                Pgis = Pgi.T
                Flis = Fli.T.to(torch.float32)
                
                load_profile[:, gen_loc] += Pgi
                nonzero_values, _ = torch.min(load_profile.masked_fill(load_profile == 0, float('inf')), dim=1, keepdim=True)
                mask = load_profile == 0
                load_profile = torch.where(mask, nonzero_values, load_profile)

                #================ compute cost and infeasibility
                infeasibility_batch = sum(lam1i.cpu().detach().numpy()) + sum(lam2i.cpu().detach().numpy())
                infeasibility_batch = torch.tensor(infeasibility_batch)
                infeasibility_tensor = lam1i + lam2i
                infeasibility_tensor[abs(infeasibility_tensor) < 1e-5 ] = 0
                infeasible_buses = (abs(infeasibility_tensor) > 1e-5)
                
                infeasible_cases = torch.zeros((batch_size,1))
                row_has_true = torch.any(infeasible_buses, axis=1)
                infeasible_cases[row_has_true] = True
                infeasible_cases = infeasible_cases.sum()
                
                batch_cost = torch.matmul(gencost,Pgis).cpu()
                batch_relco = ((batch_cost-cost_scopf_batch)/cost_scopf_batch)*100

                #===================== compute contingency flows for each sample
                imbalancetest_batch = 0
                count_crit_violation = 0
                count_line_violation = 0
                count_violation = 0

                for b in (range(tot_lodf_batches)):
                    lodf_batch = lodf_batch_size
                    if b == (tot_lodf_batches - 1): 
                        lodf_batch = last_batch_size
                    left_ones = torch.ones((lodf_batch_size,1))
                    right_ones = torch.ones((int(l*batch_size),1))
                    Fli_0 = Flis.expand(lodf_batch, -1, -1)#.contiguous().view(-1, len(Flis[1])).detach()#.to(torch.float32).clone()
                    F_max = max_f.expand(lodf_batch, -1, -1)

                    index1 = LODF_dict[b][0].detach()
                    index2 = LODF_dict[b][1].detach()
                    indices = torch.stack((index1,index2)).detach()
                    values = LODF_dict[b][2].detach()
                    shape = (l,int(l*lodf_batch))
                    lodf = torch.sparse_coo_tensor(indices,values,size = shape).T.to(device)

                    imbalancetest_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf,Flis).reshape((lodf_batch,l,batch_size)))- F_max))
                    imbalancetest_batch_loop[abs(imbalancetest_batch_loop) < 1e-5] = 0
                    
                    count_crit_violation_loop = torch.sum(imbalancetest_batch_loop > tollerance_crit*max_f, dim=1)
                    count_line_violation_loop = torch.sum(imbalancetest_batch_loop > 0, dim=1)
                    imbalancetest_batch_loop = imbalancetest_batch_loop.reshape(lodf_batch,int(l*batch_size))
                    count_violation_loop = torch.count_nonzero(imbalancetest_batch_loop)
                
                    imbalancetest_batch += torch.sum(imbalancetest_batch_loop)/int(batch_size)
                
                    count_violation += count_violation_loop
                    count_crit_violation += torch.count_nonzero(count_crit_violation_loop.detach())
                    count_line_violation += torch.count_nonzero(count_line_violation_loop.detach())
        
                relcotest_epoch[i,:] = batch_relco.detach()
                cost_epoch[i,:] = batch_cost.detach()
                imbalancetest_epoch[i,:] = imbalancetest_batch.detach()
                infeasibilitytest_epoch[i,:] = infeasibility_batch.detach()
                
                infeasibility_count[i,:] = infeasible_cases.detach()
                line_violation_count[i,:] = count_line_violation.detach()
                crit_violation_count[i,:] = count_crit_violation.detach()
                #avg_violation_list[i,:] = avg_violation.detach()
                count_violation_list[i,:] = count_violation
            

    imbalancetest = torch.mean(imbalancetest_epoch).detach()

    infeasibilitytest = torch.mean(infeasibilitytest_epoch).detach()

    relcotest = torch.mean(relcotest_epoch).detach()
    costtest = torch.mean(cost_epoch).detach()
    
    #====== convergence criteria
    infeasible_cases_tot = torch.sum(infeasibility_count).detach()
    line_violation_tot = torch.sum(line_violation_count).detach()
    crit_violation_tot = torch.sum(crit_violation_count).detach()
    #tot_avg_violation = torch.sum(avg_violation_list).detach()
    count_violation_tot = torch.sum(count_violation_list).detach()
    tot_flows = l*Ncontingencies*batch_size
    
    error[epoch,:] = imbalancetest.detach().numpy()
    avg_infeasibility[epoch,:] =  infeasibilitytest.detach().numpy()
    relcost[epoch,:] =  relcotest.detach().numpy()
    cost[epoch,:] =  costtest.detach().numpy()
    end_test = time.time()
    time_test = end_test-start_test

    print(epoch, " | ", round(time_test,0), " | ",np.round(error[epoch,:],3), " | ", np.round(torch.mean(clil.cpu()).detach().numpy(),3), " | ", np.round(avg_infeasibility[epoch,:],3), " | ", np.round(relcost[epoch,:], 3) )
        

print('method time:', time_test)

print('current infeasibility:', (infeasible_cases_tot/Nsamples)*100, '%', 'goal:', 1, '%')
    
#print('current avg violation:', (tot_avg_violation/initial_violation_count),'pu', 'goal', 0.01*max_f, 'pu')
print('infeasible contingency cases:', line_violation_tot, (line_violation_tot/(Ncontingencies*Nsamples))*100, '%')
print('10%+ infeasible cases:', crit_violation_tot, (crit_violation_tot/(Ncontingencies*Nsamples))*100, '%')
print('number of violations:', count_violation_tot)
print('total cost:', cost.sum())

#==============================================

print('------------------- heuristic -----------------------------')
#============== Create new training data (with high variability from RES)
Nsamples = 1000
#X = np.array(data['load']['p'])/Sbase*ls.kumaraswamymontecarlo(1.6, 2.8, 0.75, LB, UB, Nsamples).T

file_path = 'Xtest2_118_spyros.pkl'  
with open(file_path, 'rb') as f:
    X = pickle.load(f)

Xtrain = torch.tensor(X[0:int(Nsamples),:],dtype = torch.float32)        
Xtrain_transpose = Xtrain.transpose(0,1)
#Xtest = torch.tensor(X[int(train*Nsamples):,:],dtype = torch.float64)
#standardise
Xmin, Xmax,Xmean,Xstd = np.min(X, axis = 0),np.max(X, axis = 0),np.mean(X, axis = 0),np.std(X, axis = 0)
Xscal = torch.tensor(( X - Xmean ) / Xstd, dtype=torch.float32)


#===============================================
#=============== Benchmark #2 heuristic approach
if case == 'N-1':
    benchmark = 'SCOPF1screening'
    
#benchmark = 'SCOPF1screening'
#lcontingencies = [0, 58, 115, 104, 105, 27, 128, 95, 60, 40, 28, 107, 50, 35, 32, 30, 125, 126, 106, 7, 37, 102, 139, 87, 140, 103, 101, 142, 141, 88, 138, 118, 137, 51, 59, 61, 62, 63, 49, 64, 54, 53]

if benchmark == 'SCOPF1screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),1)))
elif benchmark == 'SCOPF2screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),2)))
elif benchmark == 'SCOPF3screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),3)))
elif benchmark == 'SCOPF4screening':
    Nk_contingencies = np.array(list(itertools.combinations(range(Nlines),4)))

Pgisscopf = torch.zeros(Ngens,int(Nsamples),dtype = torch.float32)
Fl0sscopf = torch.zeros(Nlines,int(Nsamples),dtype = torch.float32)
#Flcs = torch.zeros(Nlines,len(Nk_contingencies),int(Nsamples),dtype = torch.float32)
lam1s = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)
lam2s = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)
#lam1sc = torch.zeros(Nbus,len(Nk_contingencies),int(Nsamples),dtype = torch.float32)
#lam2sc = torch.zeros(Nbus,len(Nk_contingencies),int(Nsamples),dtype = torch.float32)
Th0 = torch.zeros(Nbus,int(Nsamples),dtype = torch.float32)

time_scopfs = time.time()

#lcontingencies = [0, 1333, 6284, 5473, 1379, 5164, 1308, 1399, 1398, 5458, 5149, 6304, 6303, 16166, 1305, 1303, 1323, 5463, 5151, 5460, 5146, 10171, 10051, 14547, 9941, 14396, 13990, 14071, 14162, 9436, 15110, 10010, 14001, 13752, 14082, 14151, 13919, 14159, 14241, 14837, 14238, 13563, 14170, 14169, 10520, 12546, 12639, 13821, 10506, 10636, 13211, 10622, 10488, 13565, 12540, 12544, 12543, 13386, 13212, 12441, 12638]
Nk_zeros = list(range(len(Nk_contingencies)))
zero_indices = [l for l in Nk_zeros if l not in lcontingencies]
print(lcontingencies)


if benchmark == 'SCOPF1screening':
    problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf1_screening(data, lcontingencies)
elif benchmark == 'SCOPF2screening':
    problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf2_screening(data, lcontingencies)
elif benchmark == 'SCOPF3screening':
    problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf3_screening(data, lcontingencies)
elif benchmark == 'SCOPF4screening':
    problem0scopf, Pd0scopf, cg0scopf, Pgiscopf, Fl0scopf, Th0scopf, Flc, lam1, lam2, lam1c, lam2c = cs.create_scopf4_screening(data, lcontingencies)

cg0scopf.value = np.array(data['gen']['cost'])
for entry in tqdm(range(int(Nsamples)),position=0, leave=True):   
    Pd0scopf.value = X[entry,:] #np.array(data['load']['p'])/Sbase    
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

cost_scopf = torch.matmul(gencost,Pgisscopf)

#============ compute Nk flows
Flis = Fl0sscopf

count_crit_violation = 0
count_line_violation = 0

for b in (range(tot_lodf_batches)):

        lodf_batch = lodf_batch_size
        if b == (tot_lodf_batches - 1):
            lodf_batch = last_batch_size
        left_ones = torch.ones((lodf_batch,1))
        right_ones = torch.ones((int(l*batch_size),1))
        Fli_0 = Flis.expand(lodf_batch, -1, -1)

        summed_tensor = torch.zeros(lodf_batch).to(device)

        index1 = LODF_dict[b][0].detach()
        index2 = LODF_dict[b][1].detach()
        indices = torch.stack((index1,index2)).detach()
        values = LODF_dict[b][2].detach()
        shape = (l,int(l*lodf_batch))
        lodf = torch.sparse_coo_tensor(indices,values,size = shape, requires_grad = False).T.to(device).detach()

        for k in range(int(Nsamples/batch_size)):
            start = k*batch_size
            end = (k+1)*batch_size
            imbalancetrain_batch_loop = (F.relu(torch.abs(Fli_0[:,:,start:end] + torch.sparse.mm(lodf,Flis[:,start:end]).reshape((lodf_batch,l,batch_size)))- max_f)).detach()

            count_crit_violation_loop = torch.sum(imbalancetrain_batch_loop > tollerance_crit*max_f, dim=1)
            count_line_violation_loop = torch.sum(imbalancetrain_batch_loop > 0, dim=1)

            count_crit_violation += torch.count_nonzero(count_crit_violation_loop)
            count_line_violation += torch.count_nonzero(count_line_violation_loop)

crit_violation_tot = count_crit_violation
line_violation_tot = count_line_violation

print('infeasible N-k contingency cases:', line_violation_tot, (line_violation_tot/(Ncontingencies*Nsamples))*100, '%')
print('10%+ contingency cases:', crit_violation_tot, (crit_violation_tot/(Ncontingencies*Nsamples))*100, '%')
    
time_scopfs = time.time() - time_scopfs
print('heuristic approach time:', time_scopfs)

##============= check infeasibility
infeasibility_base = (lam1s + lam2s).permute(1,0)
infeasibility_base[abs(infeasibility_base) < 1e-4] = 0
infeasible_buses_base = (abs(infeasibility_base) > 1e-4) #0.001*load_profile) # 10% of total load available for generation/load shedding
scopf_cost_average = cost_scopf.mean()

infeasible_cases_base = torch.zeros((Nsamples,1))
row_has_true_base = torch.any(infeasible_buses_base, axis=1)
infeasible_cases_base[row_has_true_base] = True
infeasible_cases_tot_base = infeasible_cases_base.sum()
print('infeasible base cases:', infeasible_cases_tot_base)
print('percentage base infeasibility:', (infeasible_cases_tot_base/(Nsamples))*100, '%')
print('total average cost:', scopf_cost_average)


#=========================================

start_test = time.time()
#============== Initialize training testing data
testing_dataset = Data.TensorDataset(Xscal, cost_scopf)
test_loader = Data.DataLoader(dataset=testing_dataset, batch_size=batch_size, shuffle=False)  # Note: shuffle=False for testing

#================ Initialize tensors
error = np.zeros(shape=(Niterations,1))
avg_infeasibility = np.zeros(shape=(Niterations,1))
relcost = np.zeros(shape=(Niterations,1))
cost = np.zeros(shape=(Niterations,1))

Pgis = torch.zeros(size = (Ngens,int(batch_size)),dtype = torch.float32)
Flis = torch.zeros(size = (Nlines,int(batch_size)),dtype = torch.float32)
clil = torch.zeros(size = (Ngens,int(batch_size)),dtype = torch.float32)
infeasibility = torch.zeros(int(batch_size), requires_grad = False)

relcotest_epoch = torch.zeros(size = (int(Nsamples/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)
cost_epoch = torch.zeros(size = (int(Nsamples/batch_size),int(batch_size)),dtype = torch.float32, requires_grad = False)

imbalancetest_epoch = torch.zeros(size = (round((Nsamples)/batch_size),1),dtype = torch.float32, requires_grad = False)

infeasibilitytest_epoch = torch.zeros(size = (int(Nsamples/batch_size),Nbus),dtype = torch.float32, requires_grad = False)
infeasibility_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
line_violation_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
crit_violation_count = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
avg_violation_list = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)
count_violation_list = torch.zeros(size = (int(Nsamples/batch_size),1),dtype = torch.float32, requires_grad = False)

print("It |   Time   |    Imbalance    | Mod out  |   Infeasibility   |  Rel cost")
for epoch in range(1):    
    with torch.no_grad():
        for i, (Xbatch, cost_scopf_batch) in enumerate(tqdm(test_loader)):
                
                model.eval() # deactivate dropout, when I changed this, testing error stayed same

                Xbatch = Xbatch.clone().detach()
                cost_scopf_batch = cost_scopf_batch.clone().detach()

                Xtest_batch = ((Xbatch.cpu() * Xstd)+Xmean).to(torch.float32).to(device)
                load_profile = torch.zeros((batch_size,Nbus)).to(device)
                load_profile[:, load_loc] = Xtest_batch

                #================== Perform forward pass
                #clil = (model(Xbatch))
                clil = (model(Xbatch)+1)/2
                
                #Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, gencost, clil, solver_args=ECOS_solver_args)
                Pgi, Fli, Thi, lam1i, lam2i = cvxpylayer0(Xtest_batch, clil, solver_args=ECOS_solver_args)
                Pgis = Pgi.T
                Flis = Fli.T.to(torch.float32)
                
                load_profile[:, gen_loc] += Pgi
                nonzero_values, _ = torch.min(load_profile.masked_fill(load_profile == 0, float('inf')), dim=1, keepdim=True)
                mask = load_profile == 0
                load_profile = torch.where(mask, nonzero_values, load_profile)

                #================ compute cost and infeasibility
                infeasibility_batch = sum(lam1i.cpu().detach().numpy()) + sum(lam2i.cpu().detach().numpy())
                infeasibility_batch = torch.tensor(infeasibility_batch)
                infeasibility_tensor = lam1i + lam2i
                infeasibility_tensor[abs(infeasibility_tensor) < 1e-5 ] = 0
                infeasible_buses = (abs(infeasibility_tensor) > 1e-5)
                
                infeasible_cases = torch.zeros((batch_size,1))
                row_has_true = torch.any(infeasible_buses, axis=1)
                infeasible_cases[row_has_true] = True
                infeasible_cases = infeasible_cases.sum()
                
                batch_cost = torch.matmul(gencost,Pgis).cpu()
                batch_relco = ((batch_cost-cost_scopf_batch)/cost_scopf_batch)*100

                #===================== compute contingency flows for each sample
                imbalancetest_batch = 0
                count_crit_violation = 0
                count_line_violation = 0
                count_violation = 0

                for b in (range(tot_lodf_batches)):
                    lodf_batch = lodf_batch_size
                    if b == (tot_lodf_batches - 1): 
                        lodf_batch = last_batch_size
                    left_ones = torch.ones((lodf_batch_size,1))
                    right_ones = torch.ones((int(l*batch_size),1))
                    Fli_0 = Flis.expand(lodf_batch, -1, -1)#.contiguous().view(-1, len(Flis[1])).detach()#.to(torch.float32).clone()
                    F_max = max_f.expand(lodf_batch, -1, -1)

                    index1 = LODF_dict[b][0].detach()
                    index2 = LODF_dict[b][1].detach()
                    indices = torch.stack((index1,index2)).detach()
                    values = LODF_dict[b][2].detach()
                    shape = (l,int(l*lodf_batch))
                    lodf = torch.sparse_coo_tensor(indices,values,size = shape).T.to(device)

                    imbalancetest_batch_loop = (F.relu(torch.abs(Fli_0 + torch.sparse.mm(lodf,Flis).reshape((lodf_batch,l,batch_size)))- F_max))
                    imbalancetest_batch_loop[abs(imbalancetest_batch_loop) < 1e-5] = 0
                    
                    count_crit_violation_loop = torch.sum(imbalancetest_batch_loop > tollerance_crit*max_f, dim=1)
                    count_line_violation_loop = torch.sum(imbalancetest_batch_loop > 0, dim=1)
                    imbalancetest_batch_loop = imbalancetest_batch_loop.reshape(lodf_batch,int(l*batch_size))
                    count_violation_loop = torch.count_nonzero(imbalancetest_batch_loop)
                
                    imbalancetest_batch += torch.sum(imbalancetest_batch_loop)/int(batch_size)
                
                    count_violation += count_violation_loop
                    count_crit_violation += torch.count_nonzero(count_crit_violation_loop.detach())
                    count_line_violation += torch.count_nonzero(count_line_violation_loop.detach())
        
                relcotest_epoch[i,:] = batch_relco.detach()
                cost_epoch[i,:] = batch_cost.detach()
                imbalancetest_epoch[i,:] = imbalancetest_batch.detach()
                infeasibilitytest_epoch[i,:] = infeasibility_batch.detach()
                
                infeasibility_count[i,:] = infeasible_cases.detach()
                line_violation_count[i,:] = count_line_violation.detach()
                crit_violation_count[i,:] = count_crit_violation.detach()
                #avg_violation_list[i,:] = avg_violation.detach()
                count_violation_list[i,:] = count_violation
            

    imbalancetest = torch.mean(imbalancetest_epoch).detach()

    infeasibilitytest = torch.mean(infeasibilitytest_epoch).detach()

    relcotest = torch.mean(relcotest_epoch).detach()
    costtest = torch.mean(cost_epoch).detach()
    
    #====== convergence criteria
    infeasible_cases_tot = torch.sum(infeasibility_count).detach()
    line_violation_tot = torch.sum(line_violation_count).detach()
    crit_violation_tot = torch.sum(crit_violation_count).detach()
    #tot_avg_violation = torch.sum(avg_violation_list).detach()
    count_violation_tot = torch.sum(count_violation_list).detach()
    tot_flows = l*Ncontingencies*batch_size
    
    error[epoch,:] = imbalancetest.detach().numpy()
    avg_infeasibility[epoch,:] =  infeasibilitytest.detach().numpy()
    relcost[epoch,:] =  relcotest.detach().numpy()
    cost[epoch,:] =  costtest.detach().numpy()
    end_test = time.time()
    time_test = end_test-start_test

    print(epoch, " | ", round(time_test,0), " | ",np.round(error[epoch,:],3), " | ", np.round(torch.mean(clil.cpu()).detach().numpy(),3), " | ", np.round(avg_infeasibility[epoch,:],3), " | ", np.round(relcost[epoch,:], 3) )
        

print('method time:', time_test)

print('current infeasibility:', (infeasible_cases_tot/Nsamples)*100, '%', 'goal:', 1, '%')
    
#print('current avg violation:', (tot_avg_violation/initial_violation_count),'pu', 'goal', 0.01*max_f, 'pu')
print('infeasible contingency cases:', line_violation_tot, (line_violation_tot/(Ncontingencies*Nsamples))*100, '%')
print('10%+ infeasible cases:', crit_violation_tot, (crit_violation_tot/(Ncontingencies*Nsamples))*100, '%')
print('number of violations:', count_violation_tot)
print('total cost:', cost.sum())





























