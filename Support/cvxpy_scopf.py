# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 08:24:41 2021

@author: jlc516
"""
import cvxpy as cp
import pandas as pd
import numpy as np
import itertools
import torch

def create_scopf_jochen(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus-1,nconti))
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ Pg >= -data['gen']['max_p']/Sbase, Pg <= -data['gen']['min_p']/Sbase ]  
    constraints += [ Fl >= -data['line']['max_f']/Sbase, Fl <= data['line']['max_f']/Sbase ] 
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    #constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    #constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    
    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and l1 != lcontingencies[c]) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and l2 != lcontingencies[c])) ]   
    
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
    for c in range(nconti):
        for l in range (nline):
            if l!=lcontingencies[c]: 
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
            
        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) )
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc

def create_scopf(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus,nconti))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti))
    lam2c = cp.Variable((nbus,nconti))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    

    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and l1 != lcontingencies[c]) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and l2 != lcontingencies[c])) ]   

    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
    for c in range(nconti):
        for l in range (nline):
            if l!=lcontingencies[c]: 
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c


def create_scopf1_screening(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus,nconti))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti))
    lam2c = cp.Variable((nbus,nconti))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    

    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and l1 != lcontingencies[c]) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and l2 != lcontingencies[c])) ]   

    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
    for c in range(nconti):
        for l in range (nline):
            if l!=lcontingencies[c]: 
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c




def create_scopf2_screening(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    N2 = np.array(list(itertools.combinations(range(nline),2)))
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus,nconti))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti))
    lam2c = cp.Variable((nbus,nconti))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ] 
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    
    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and (l1 != N2[lcontingencies[c],0] and l1 != N2[lcontingencies[c],1])) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and (l2 != N2[lcontingencies[c],0] and l2 != N2[lcontingencies[c],1]))) ]   

    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
        
    for c in range(nconti):
        for l in range (nline):
            if (l != N2[lcontingencies[c],0] and l != N2[lcontingencies[c],1]):
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c



def create_scopf_risk_based(data,lcontingencies,l2contingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    nconti2 = len(l2contingencies)
    N2 = np.array(list(itertools.combinations(range(nline),2)))
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    
    Flc = cp.Variable((nline,nconti+nconti2))
    Thc = cp.Variable((nbus,nconti+nconti2))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti+nconti2))
    lam2c = cp.Variable((nbus,nconti+nconti2))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ]
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),((nconti+nconti2),1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,((nconti+nconti2),1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    

    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and l1 != lcontingencies[c]) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and l2 != lcontingencies[c])) ]   
    
    counter = 0
    for c in range(nconti, nconti+nconti2):        
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and l1 != N2[l2contingencies[counter],0] and l1 != N2[l2contingencies[counter],1]) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and l2 != N2[l2contingencies[counter],0] and l2 != N2[l2contingencies[counter],1])) ]   
        counter += 1
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
    
    for c in range(nconti):
        for l in range (nline):
            if l!=lcontingencies[c]: 
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

    counter = 0
    for c in range(nconti, nconti+nconti2):
        for l in range (nline):
            if l != N2[l2contingencies[counter],0] and l != N2[l2contingencies[counter],1]: 
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
        counter += 1
        
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c



def create_scopf3_screening(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    N3 = np.array(list(itertools.combinations(range(nline),3)))
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus,nconti))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti))
    lam2c = cp.Variable((nbus,nconti))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ] 
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    
    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and (l1 != N3[lcontingencies[c],0] and l1 != N3[lcontingencies[c],1] and l1 != N3[lcontingencies[c],2])) )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and (l2 != N3[lcontingencies[c],0] and l2 != N3[lcontingencies[c],1] and l2 != N3[lcontingencies[c],2]))) ]   

    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
        
    for c in range(nconti):
        for l in range (nline):
            if (l != N3[lcontingencies[c],0] and l != N3[lcontingencies[c],1] and l != N3[lcontingencies[c],2]):
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c



def create_scopf4_screening(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    N4 = np.array(list(itertools.combinations(range(nline),4)))
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus,nconti))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti))
    lam2c = cp.Variable((nbus,nconti))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ] 
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    
    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and (l1 != N4[lcontingencies[c],0] and l1 != N4[lcontingencies[c],1] and l1 != N4[lcontingencies[c],2] and l1 != N4[lcontingencies[c],3]) ))  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and (l2 != N4[lcontingencies[c],0] and l2 != N4[lcontingencies[c],1] and l2 != N4[lcontingencies[c],2] and l2 != N4[lcontingencies[c],3]))) ]   

    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
        
    for c in range(nconti):
        for l in range (nline):
            if (l != N4[lcontingencies[c],0] and l != N4[lcontingencies[c],1] and l != N4[lcontingencies[c],2] and l != N4[lcontingencies[c],3]):
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c





def create_scopf5_screening(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = len(lcontingencies)
    N5 = np.array(list(itertools.combinations(range(nline),5)))
    
    b_a = np.arange(nbus)
    #b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus)
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus,nconti))
    lam1 = cp.Variable(nbus)
    lam2 = cp.Variable(nbus)
    lam1c = cp.Variable((nbus,nconti))
    lam2c = cp.Variable((nbus,nconti))
    
    minfl = torch.tensor(-data['line']['max_f']/Sbase)
    maxfl = torch.tensor(data['line']['max_f']/Sbase)
    ming = torch.tensor(-data['gen']['max_p']/Sbase)
    maxg = torch.tensor(-data['gen']['min_p']/Sbase)
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ lam1 >= 0.0, lam2 >=0.0, lam1c >= 0.0, lam2c >=0.0]
    constraints += [ Pg >= ming , Pg <= maxg ]
    constraints += [ Fl >= minfl, Fl <= maxfl ] 
    constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == lam1[b] - lam2[b] + sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    
    for c in range(nconti):
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == lam1c[b,c] - lam2c[b,c] + sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and (l1 != N5[lcontingencies[c],0] and l1 != N5[lcontingencies[c],1] and l1 != N5[lcontingencies[c],2] and l1 != N5[lcontingencies[c],3] and l1 != N5[lcontingencies[c],4]) ))  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and (l2 != N5[lcontingencies[c],0] and l2 != N5[lcontingencies[c],1] and l2 != N5[lcontingencies[c],2] and l2 != N5[lcontingencies[c],3] and l2 != N5[lcontingencies[c],4]))) ]   

    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
        
    for c in range(nconti):
        for l in range (nline):
            if (l != N5[lcontingencies[c],0] and l != N5[lcontingencies[c],1] and l != N5[lcontingencies[c],2] and l != N5[lcontingencies[c],3] and l != N5[lcontingencies[c],4]):
                constraints += [ Flc[l,c] == Sbase/data['line']['x'][l] *(sum(Thc[k1,c] for k1 in range(nbus) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus) if int(data['line']['to_bus'][l])==b_a[k2])) ]

    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)) + 1000000* cp.sum(lam1) + 999999* cp.sum(lam2) + 1000* cp.sum(lam1c) + 999* cp.sum(lam2c))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc, lam1, lam2, lam1c, lam2c







def create_scopf2(data,lcontingencies): 
    #PREPARE OPTIMIZATION PROBLEM
    #Variables and parameter declarations
    ngen = len(data['gen'])
    nload = len(data['load'])
    nbus = len(data['bus'])
    nline = len(data['line'])
    Sbase = data['par']['base'][0]
    
    nconti = lcontingencies #len(lcontingencies)
    N2 = np.array(list(itertools.combinations(range(nline),2)))
    
    b_a = np.arange(nbus)
    b_a = np.delete(b_a,data['par']['refnode'][0])
    
    Pd = cp.Parameter(nload)
    cg = cp.Parameter(ngen)
    Pg = cp.Variable(ngen)
    Fl = cp.Variable(nline)
    Th = cp.Variable(nbus-1)
    
    Flc = cp.Variable((nline,nconti))
    Thc = cp.Variable((nbus-1,nconti))
    
    #Pd.value =  np.array(data['load']['p'])/Sbase
    #cg.value = np.array(data['gen']['cost'])
    
    #Constraints
    constraints = [ Pg >= -data['gen']['max_p']/Sbase, Pg <= -data['gen']['min_p']/Sbase ]  
    constraints += [ Fl >= -1.5*data['line']['max_f']/Sbase, Fl <= 1.5*data['line']['max_f']/Sbase ] 
    #constraints += [Th[data['par']['refnode'][0]] == data['par']['va_degree'][0]]
    
    ####constraints += [ Flc >= np.tile((-data['line']['max_f']/Sbase),(nconti,1)).transpose() , Flc <= np.tile(data['line']['max_f']/Sbase,(nconti,1)).transpose() ]
    #constraints += [Thc[data['par']['refnode'][0],:] == data['par']['va_degree'][0]]
    
    #constraint for node balances
    for b in range(nbus):
        constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2]) == b )
                        == sum(Fl[l1] for l1 in range(nline) if int( data['line']['to_bus'][l1]) == b )  - sum(Fl[l2] for l2 in range(nline) if int(data['line']['from_bus'][l2]) == b ) ]
    
    for c in lcontingencies:
        for b in range(nbus):
            constraints += [ - sum(Pd[k1] for k1 in range(nload) if int( data['load']['bus'][k1]) == b ) + sum(Pg[k2] for k2 in range(ngen) if int(data['gen']['bus'][k2])==b) 
                            == sum(Flc[l1,c] for l1 in range(nline) if ( int( data['line']['to_bus'][l1]) == b and l1 != N2[c,0]) or N2[c,1] )  - sum(Flc[l2,c] for l2 in range(nline) if (int(data['line']['from_bus'][l2])==b and l2 != N2[c,0] or N2[c,1])) ]   
    
    
    #constraint for line flows
    for l in range (nline):
        constraints += [ Fl[l] == Sbase/data['line']['x'][l] *(sum(Th[k1] for k1 in range(nbus-1) if int(data['line']['from_bus'][l])==b_a[k1]) - sum(Th[k2] for k2 in range(nbus-1) if int(data['line']['to_bus'][l])==b_a[k2])) ]
    
    for c in lcontingencies:
        for l in range (nline):
            for ll in range (l+1,nline):
                if l!=lcontingencies[c]:
                    if ll!= lcontingencies[c]: 
                        constraints += [ Flc[ll,c] == Sbase/data['line']['x'][ll] *(sum(Thc[k1,c] for k1 in range(nbus-1) if int(data['line']['from_bus'][ll])==b_a[k1]) - sum(Thc[k2,c] for k2 in range(nbus-1) if int(data['line']['to_bus'][ll])==b_a[k2])) ]
                
    
    objective = cp.Minimize( cp.sum( cp.multiply(cg,Pg)))
                    
    problem= cp.Problem(objective, constraints)
    
    return problem, Pd, cg, Pg, Fl, Th, Flc

# # #USE:
# data = pd.read_excel('IEEE39.xlsx',sheet_name=None)
# Sbase = data['par']['base'][0]

# #Set to DCOPF solution
# problem, Pd, cg, Pg, Fl, Th = create_scopf(data,lcontingencies = range(len(data['line'])))
# Pd.value = np.array(data['load']['p'])/Sbase
# cg.value = np.array(data['gen']['cost'])
# solution  = problem.solve(solver=cp.MOSEK)
# if problem.status not in ["infeasible", "unbounded"]:
#     # Otherwise, problem.value is inf or -inf, respectively.
#     print("Optimal value: %s" % problem.value)