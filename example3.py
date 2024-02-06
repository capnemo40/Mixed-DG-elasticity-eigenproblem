from netgen.geom2d import SplineGeometry
from ngsolve import *

import numpy as np

'''
Code corresponding to  the Numerical Results (Example 3) of

@article{MEDDAHI202319,
title = {A DG method for a stress formulation of the elasticity 
eigenproblem with strongly imposed symmetry},
journal = {Computers & Mathematics with Applications},
volume = {135},
pages = {19-30},
year = {2023},
doi = {https://doi.org/10.1016/j.camwa.2023.01.022},
author = {Salim Meddahi}
}

Eigenvalues of the elasticity system
Unit disk,  Pure displacement BCs
Nearly incompressible case
DG approximation solely in terms of stress with strong symmetry
Use known Stokes eigenvalues to generate the errors in Table 6.9
'''

def eigenweaksym(k, h):
    
    # Mesh of the unit disk
    geo = SplineGeometry()
    geo.AddCircle( (0,0), 1, bc="bnd")
    mesh = Mesh(geo.GenerateMesh (maxh=h))
    mesh.Curve(10)
    
    # Elastic component
    rho = 1
    nu =0.5 - 1e-12
    E = 1

    mu = (1 / 2) * (E / (1 + nu)) 
    landa = E * nu / ((1 + nu) * (1 - 2 * nu))
    
    a1 = 0.5 / mu
    a2 = landa / (4.0 * mu * (landa + mu))
    
    sota = 8 #DG stability parameter
    
    
    # ********* Finite dimensional spaces ********* #

    S = L2(mesh, order =k, dgjumps=True, complex=True)#, complex=True)
    fes = FESpace([S, S, S], dgjumps=True, complex=True)
    
    # ********* test and trial functions for product space ****** #
    sigma1, sigma12, sigma2 = fes.TrialFunction()
    tau1,   tau12,   tau2   = fes.TestFunction()
    
    sigma = CoefficientFunction(( sigma1, sigma12, sigma12, sigma2), dims = (2,2) )
    tau   = CoefficientFunction(( tau1,   tau12,   tau12,   tau2),   dims = (2,2) )
  
    Asig = a1 * sigma - a2 * Trace(sigma) *  Id(mesh.dim)
       
    divsigma = CoefficientFunction( ( grad(sigma1)[0] + grad(sigma12)[1], grad(sigma12)[0] + grad(sigma2)[1]) )
    divtau   = CoefficientFunction( ( grad(tau1)[0] + grad(tau12)[1], grad(tau12)[0] + grad(tau2)[1]) )
    
    n = specialcf.normal(mesh.dim)
    jump_sigma = sigma*n - CoefficientFunction( (sigma1.Other()*n[0] + sigma12.Other()*n[1], 
                                                 sigma12.Other()*n[0] + sigma2.Other()*n[1] ) )
    jump_tau   = tau*n - CoefficientFunction( (tau1.Other()*n[0] + tau12.Other()*n[1], 
                                               tau12.Other()*n[0] + tau2.Other()*n[1] ) )
    
    mean_divsigma = 0.5*( divsigma + CoefficientFunction( ( grad(sigma1.Other())[0] 
                                                           + grad(sigma12.Other())[1], grad(sigma12.Other())[0] + grad(sigma2.Other())[1]) ) )
    mean_divtau   = 0.5*( divtau   + CoefficientFunction( ( grad(tau1.Other())[0]   
                                                           + grad(tau12.Other())[1],   grad(tau12.Other())[0]   + grad(tau2.Other())[1]) ) )                   
    h = specialcf.mesh_size

    # ********* bilinear forms ****** #
    
    a = BilinearForm (fes, symmetric=True)    
    a +=  InnerProduct(divsigma, divtau)*dx + InnerProduct(Asig, tau)*dx   
    a += sota*k**2/h*InnerProduct(jump_sigma, jump_tau) * dx(skeleton=True )
    a += -(mean_divsigma*jump_tau + mean_divtau*jump_sigma) * dx(skeleton=True )
    
    m = BilinearForm (fes, symmetric=True)
    m += InnerProduct(Asig, tau)*dx
    
    a.Assemble()
    m.Assemble()
    
    # ********* solving the eigenproblem ****** #
    u = GridFunction(fes, multidim=15)
    
    with TaskManager():
        lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), 
                            list(u.vecs), shift=10.5)
        
    eigen = []
    for evals in lam:
        if evals.real-1 > 1: eigen.append(3*(evals.real-1))
        
    eigen = np.sort(eigen[0:5])
    
    return eigen

# ********* eigenvalues collector ****** #

def collecterrors(k, maxr):
    eigen_list = []
    
    for l in range(0, maxr):
        hl = 2 ** (-l) / 2
        lam = eigenweaksym(k, hl)
        eigen_list.append(lam)

    eigen_list = np.array(eigen_list)

    return eigen_list

# ********* convergence table ****** #

def hconvergencetable(eigen_list, eigenExact, maxr):
    print("===============================================================================")
    print(" Mesh   Eigenvalue1   Eigenvalue2    Eigenvalue3   Eigenvalue4   Eigenvalue5   ")
    print("-------------------------------------------------------------------------------")

    rates = np.zeros((maxr, 5))

    for j in range(0, 5):
        for i in range(1, maxr):

            e1 = abs( eigenExact[j] - eigen_list[i-1, j] )
            e2 = abs( eigenExact[j] - eigen_list[i, j] )

            if abs( e1) > 1.e-15:
                rates[i,j] = format(log( e1 / e2 ) / log(2), '+5.2f')

    for i in range(maxr):
        print(" h/%-4d  %3.12f   %3.12f   %3.12f   %3.12f  %3.12f  " 
            %(2**(i + 2), 
                eigen_list[i,0], eigen_list[i,1], eigen_list[i,2], eigen_list[i,3], 
                    eigen_list[i,4] ))
    
    print("  exact  %3.12f   %3.12f   %3.12f   %3.12f  %3.12f " 
        % (eigenExact[0], eigenExact[1], eigenExact[2], eigenExact[3], eigenExact[4] ) )
    
    print("===============================================================================")
    print("Rates of convergence")
    print("-------------------------------------------------------------------------------")

    for i in range(maxr):
        print(" h/%-4d  %3.4f   %3.4f   %3.4f   %3.4f  %3.4f  " 
            %(2**(i + 2), 
                rates[i,0], rates[i,1], rates[i,2], rates[i,3], rates[i,4] ))
           
    print("===================================================================================")

    return rates



# ********* exact eigenvalues ****** #
eigenExact = np.array([ 14.681970642124,   26.3746164271634,   26.3746164271634, 40.7064658182003,   
   40.7064658182003])#, 49.218456321695,  57.5829409032911])#,  70.8499989190959, 76.9389283336474])


maxlevels = 4 #levels of mesh refinements
k = 4 #polynomial degree

eigen_list = collecterrors(k, maxlevels)

rates = hconvergencetable( eigen_list, eigenExact,  maxlevels)