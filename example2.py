from ngsolve import *
import numpy as np
from netgen.read_gmsh import ReadGmsh

'''
Code corresponding to  the Numerical Results (Example 2) of

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
Unit cube,  pure displacement BCs
DG approximation solely in terms of stress with strong symmetry

Table 6.7: k=2, h = 1/4, sota = 4, 8, 16
Table 6.8: k=3,4, h = 1/4, sota = 8
'''

# read the mesh of the unit cube with barycentric refinement 
ngmesh = ReadGmsh("cubeBary.msh")
mesh = ngsolve.Mesh(ngmesh)

# ********* Model coefficients and parameters ********* #

k = 2   #polynomial order
sota = 8 #DG stability parameter

rho = 1
nu =0.35
E = 1

mu = 1 / 2 * (E / (1 + nu)) 
landa = E * nu / ((1 + nu) * (1 - 2 * nu))

a1 = 0.5 / mu                 
a2 = landa/ (2.0 * mu * (3 * landa + 2 * mu))

# ********* Finite dimensional spaces ********* #

S = L2(mesh, order =k, dgjumps=True)
fes = FESpace([S, S, S, S, S, S], dgjumps=True)

# ********* test and trial functions for product space ****** #

sigma1, sigma12, sigma13, sigma2, sigma23, sigma3 = fes.TrialFunction()
tau1,   tau12,   tau13,   tau2,   tau23,   tau3   = fes.TestFunction()

sigma = CoefficientFunction(( sigma1, sigma12, sigma13, \
    sigma12, sigma2, sigma23,\
        sigma13, sigma23, sigma3), dims = (3,3) )
tau   = CoefficientFunction(( tau1,   tau12,   tau13,\
    tau12,   tau2,   tau23,\
        tau13,   tau23,   tau3),   dims = (3,3) )

Asig = a1 * sigma - a2 * Trace(sigma) *  Id(mesh.dim)

divsigma = CoefficientFunction( ( grad(sigma1)[0] + grad(sigma12)[1] + grad(sigma13)[2],\
    grad(sigma12)[0] + grad(sigma2)[1] + grad(sigma23)[2],\
        grad(sigma13)[0] + grad(sigma23)[1] + grad(sigma3)[2]) ) 
divtau   = CoefficientFunction( ( grad(tau1)[0] + grad(tau12)[1] + grad(tau13)[2],\
    grad(tau12)[0] + grad(tau2)[1] + grad(tau23)[2],\
        grad(tau13)[0] + grad(tau23)[1] + grad(tau3)[2]) )

n = specialcf.normal(mesh.dim)
jump_sigma = sigma*n - CoefficientFunction( (sigma1.Other()*n[0] + sigma12.Other()*n[1] + sigma13.Other()*n[2],\
    sigma12.Other()*n[0] + sigma2.Other()*n[1] +  sigma23.Other()*n[2],\
        sigma13.Other()*n[0] + sigma23.Other()*n[1] + sigma3.Other()*n[2]) )
jump_tau   = tau*n - CoefficientFunction( (tau1.Other()*n[0] + tau12.Other()*n[1] + tau13.Other()*n[2],\
    tau12.Other()*n[0] + tau2.Other()*n[1] +  tau23.Other()*n[2],\
        tau13.Other()*n[0] + tau23.Other()*n[1] + tau3.Other()*n[2]) )

mean_divsigma = 0.5*( divsigma + CoefficientFunction( ( grad(sigma1.Other())[0] + grad(sigma12.Other())[1] + grad(sigma13.Other())[2] , \
    grad(sigma12.Other())[0] + grad(sigma2.Other())[1] + grad(sigma23.Other())[2],\
        grad(sigma13.Other())[0] + grad(sigma23.Other())[1] + grad(sigma3.Other())[2] ) ) )
mean_divtau   = 0.5*( divtau + CoefficientFunction( ( grad(tau1.Other())[0] + grad(tau12.Other())[1] + grad(tau13.Other())[2] , \
    grad(tau12.Other())[0] + grad(tau2.Other())[1] + grad(tau23.Other())[2],\
        grad(tau13.Other())[0] + grad(tau23.Other())[1] + grad(tau3.Other())[2] ) ) )              



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

u = GridFunction(fes, multidim=20, name='resonances')

with TaskManager():
    lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), 
                        list(u.vecs), shift= 4)


eigen = []
for evals in lam:
    if evals.real > 1.01: eigen.append(np.sqrt(evals.real - 1))

# ********* listing the eigenvalues ****** #

for i in range(len(eigen)):
        print("%1.6f"  %(eigen[i]))