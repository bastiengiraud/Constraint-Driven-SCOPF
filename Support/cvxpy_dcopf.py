# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:24:41 2021

@author: jlc516
"""
import cvxpy as cp
import pandas as pd
import numpy as np
import torch


def create_dcopf_upper(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming , Pg <= ming + cp.multiply((maxg - ming),cl) ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000* cp.sum(lam1) + 999* cp.sum(lam2))
                    
    problemDCOPF= cp.Problem(objective, constraints)

    
    return problemDCOPF, Pd, cg, cl, Pg, Fl, Th, lam1, lam2

def create_dcopf_lower(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming + cp.multiply((maxg - ming),cl), Pg <= maxg  ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000*cp.sum(lam1) + 999*cp.sum(lam2))
                    
    problemDCOPF= cp.Problem(objective, constraints)

    
    return problemDCOPF, Pd, cg, cl, Pg, Fl, Th, lam1, lam2


def create_dcopf_correction(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    #cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming, Pg <= maxg  ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    #objective = cp.Minimize( 10*cp.sum(cp.abs(Pg - (ming + cp.multiply((maxg - ming),cl)))) + 1000*cp.sum(lam1) + 999*cp.sum(lam2)) 
    objective = cp.Minimize( cp.sum(cp.norm(Pg - (ming + cp.multiply((maxg - ming),cl)), 2)) + 1000*cp.sum(lam1) + 999*cp.sum(lam2)) 
    # can't apply jacobian with quadratic objective error
                    
    problemDCOPF= cp.Problem(objective, constraints)
    
    
    return problemDCOPF, Pd, cl, Pg, Fl, Th, lam1, lam2

def create_dcopf_correction0(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    #cg = cp.Parameter(ngen)
    #cl = cp.Parameter(ngen)
    Pg = cp.Parameter(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming, Pg <= maxg  ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    #objective = cp.Minimize( 10*cp.sum(cp.abs(Pg - (ming + cp.multiply((maxg - ming),cl)))) + 1000*cp.sum(lam1) + 999*cp.sum(lam2)) 
    objective = cp.Minimize(  1000*cp.sum(lam1) + 999*cp.sum(lam2)) 
    # can't apply jacobian with quadratic objective error
                    
    problemDCOPF= cp.Problem(objective, constraints)
    
    
    return problemDCOPF, Pd, Pg, Fl, Th, lam1, lam2


def create_dcopf_correction2(data, gen_fixed, gen_variable): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(gen_variable)
    ngen_fix = len(gen_fixed)
    #ngen_fixed = len(data['gen_fixed'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    #cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Pg_fixed = cp.Parameter(ngen_fix)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-(gen_variable['max_p'].values)/Sbase)
    maxg = torch.tensor(-(gen_variable['min_p'].values)/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming, Pg <= maxg  ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int((gen_variable['bus'].values)[k2])==b) + sum(Pg_fixed[k3] for k3 in range(ngen_fix) if int((gen_fixed['bus'].values)[k3])==b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    objective = cp.Minimize( cp.sum(cp.norm(Pg - (ming + cp.multiply((maxg - ming),cl)), 2)) + 1000*cp.sum(lam1) + 999*cp.sum(lam2)) 
                    
    problemDCOPF= cp.Problem(objective, constraints)
    
    
    return problemDCOPF, Pd, cl, Pg, Pg_fixed, Fl, Th, lam1, lam2

def create_dcopf_correction3(data, gen_fixed, gen_variable, load_fixed, load_input): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(gen_variable)
    ngen_fix = len(gen_fixed)
    #ngen_fixed = len(data['gen_fixed'])
    nload = len(load_input)
    nload_fix = len(load_fixed)
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    Pd_fixed = cp.Parameter(nload_fix)
    #cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Pg_fixed = cp.Parameter(ngen_fix)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-(gen_variable['max_p'].values)/Sbase)
    maxg = torch.tensor(-(gen_variable['min_p'].values)/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming, Pg <= maxg  ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd_fixed[k0] for k0 in range(nload_fix) if int( (load_fixed['bus'].values)[k0]) == b ) - sum(Pd[k1] for k1 in range(nload) if int( (load_input['bus'].values)[k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int((gen_variable['bus'].values)[k2])==b) + sum(Pg_fixed[k3] for k3 in range(ngen_fix) if int((gen_fixed['bus'].values)[k3])==b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    objective = cp.Minimize( cp.sum(cp.norm(Pg - (ming + cp.multiply((maxg - ming),cl)), 2)) + 1000*cp.sum(lam1) + 999*cp.sum(lam2)) 
                    
    problemDCOPF= cp.Problem(objective, constraints)
    
    
    return problemDCOPF, Pd, Pd_fixed, cl, Pg, Pg_fixed, Fl, Th, lam1, lam2



def create_dcopf_both(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    cl_low = cp.Parameter(ngen)
    cl_high = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #initialize
    #cl_low.value = np.zeros((ngen))
    #cl_high.value = np.ones((ngen))
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming + cp.multiply((maxg - ming),(cl_low)), Pg <= maxg - cp.multiply((maxg - ming),(cl_high)) ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000* cp.sum(lam1) + 999* cp.sum(lam2))
                    
    problemDCOPF= cp.Problem(objective, constraints)

    
    return problemDCOPF, Pd, cg, cl_low, cl_high, Pg, Fl, Th, lam1, lam2


def create_dcopf_direct(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    #constraints += [cl >= np.zeros(ngen), cl <= np.ones(ngen)]
    constraints += [Pg == ming + cp.multiply((maxg - ming),cl)]
    
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000* cp.sum(lam1) + 999* cp.sum(lam2))
                    
    problemDCOPF= cp.Problem(objective, constraints)

    
    return problemDCOPF, Pd, cg, cl, Pg, Fl, Th, lam1, lam2

def create_dcopf_slot(data): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    cl = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    subdomain = 0.1
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, Pg >=0.0]  
    constraints += [ Pg >= ming + (1-subdomain)*cp.multiply((maxg - ming),cl), Pg <= subdomain + ming + (1-subdomain)*cp.multiply((maxg - ming),cl)  ]
    
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b)]
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000* cp.sum(lam1) + 999* cp.sum(lam2))
                    
    problemDCOPF= cp.Problem(objective, constraints)

    
    return problemDCOPF, Pd, cg, cl, Pg, Fl, Th, lam1, lam2

def create_dcopf_many(data,ndata): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    Pd = cp.Parameter(shape=(nload,ndata))
    cg = cp.Variable(shape = (ngen,ndata))
    Pg = cp.Variable(shape = (ngen,ndata))
    Fl = cp.Variable(shape=(nline,ndata))
    Th = cp.Variable(shape = (nbus-1,ndata))
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    maxp = np.repeat(np.array(data['gen']['max_p'])[:,np.newaxis]/Sbase, ndata, 1)
    minp = np.repeat(np.array(data['gen']['min_p'])[:,np.newaxis]/Sbase, ndata, 1)
    maxf = np.repeat(np.array(data['line']['max_f'])[:,np.newaxis]/Sbase, ndata, 1)

    
    #Constraints
    constraints = [ Pg >= -maxp, Pg <= -minp ]
    constraints += [ Fl >= -maxf, Fl <= maxf ]
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    
    for n in range(ndata):
        #constraint for node balances
        for b in range(nbus):
            constraints += [ - sum(Pd[k1,n] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2,n] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == sum(Fl[l1,n] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2,n] for l2 in range(nline) if int(data['line']['from_bus'][l2])==b) ]   
        
        #constraint for line flows
        for l in range (nline):
            constraints += [ Fl[l,n] == Sbase/data['line']['x'][l] *(sum(Th[k1,n] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2,n] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        
    
    objective = cp.Minimize( cp.sum(cp.sum( cp.multiply(cg,Pg))))
    
    #objective = cp.Minimize( cp.sum(cp.sum_squares(cg-Pg)))
     
   
    problemDCOPF= cp.Problem(objective, constraints)

    
    return problemDCOPF, Pd, cg, Pg, Fl, Th

# =============================================================================
# #USE:
# data = pd.read_excel('IEEE39.xlsx',sheet_name=None)
# Sbase = data['par']['base'][0]
# #Set to DCOPF solution
# problem, Pd, cg, Pg, Fl, Th = create_dcopf(data)
# Pd.value = np.array(data['load']['p'])/Sbase
# cg.value = np.array(data['gen']['cost'])
# solution  = problem.solve(solver=cp.MOSEK)
# if problem.status not in ["infeasible", "unbounded"]:
#     # Otherwise, problem.value is inf or -inf, respectively.
#     print("Optimal value: %s" % problem.value)
# =============================================================================