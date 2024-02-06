from netgen.geom2d import unit_square
from ngsolve import *

'''
Code corresponding to  the Numerical Results (Example 1) of

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
Unit square,  mixed BCs
DG approximation solely in terms of stress with strong symmetry

Table 6.2: k=1, h = 1/16, sota = 2, 4, 8
Table 6.4: k=2, h = 1/16, sota = 2, 4, 8
Table 6.5: k=3, h = 1/16, sota = 2, 4, 8
Table 6.6: k=4, h = 1/16, sota = 2, 4, 8
'''

# ********* Model coefficients and parameters ********* #

k = 2
h = 1/16
sota = 8 #DG stability parameter


# Mesh of the unit square
mesh = Mesh(unit_square.GenerateMesh(maxh=h))

if k <= 2: #Performing barycentric refinements if k <= 2
  mesh.SplitElements_Alfeld()


# Elastic component
rho = 1
nu =0.35
E = 1

mu = 1 / 2 * (E / (1 + nu)) 
landa = E * nu / ((1 + nu) * (1 - 2 * nu))

a1 = 0.5 / mu
a2 = landa / (4.0 * mu * (landa + mu))

# ********* Finite dimensional spaces ********* #

S = L2(mesh, order =k, dgjumps=True)
fes = FESpace([S, S, S], dgjumps=True)

# ********* test and trial functions for product space ****** #

sigma1, sigma12, sigma2 = fes.TrialFunction()
tau1, tau12, tau2 = fes.TestFunction()

sigma = CoefficientFunction(( sigma1, sigma12, sigma12, sigma2), dims = (2,2) )
tau   = CoefficientFunction(( tau1,   tau12,   tau12,   tau2),   dims = (2,2) )

Asig = a1 * sigma - a2 * Trace(sigma) *  Id(mesh.dim)

divsigma = CoefficientFunction( ( grad(sigma1)[0] + grad(sigma12)[1], 
                                 grad(sigma12)[0] + grad(sigma2)[1]) )
divtau   = CoefficientFunction( ( grad(tau1)[0] + grad(tau12)[1], grad(tau12)[0]
                                 + grad(tau2)[1]) )


n = specialcf.normal(mesh.dim)
jump_sigma = sigma*n - CoefficientFunction( (sigma1.Other()*n[0] 
                                             + sigma12.Other()*n[1], sigma12.Other()*n[0] 
                                             + sigma2.Other()*n[1] ) )
jump_tau   = tau*n - CoefficientFunction( (tau1.Other()*n[0] 
                                           + tau12.Other()*n[1], tau12.Other()*n[0] 
                                           + tau2.Other()*n[1] ) )

mean_divsigma = 0.5*( divsigma + CoefficientFunction( ( grad(sigma1.Other())[0] 
                                                       + grad(sigma12.Other())[1], grad(sigma12.Other())[0] 
                                                        + grad(sigma2.Other())[1]) ) )
mean_divtau   = 0.5*( divtau   + CoefficientFunction( ( grad(tau1.Other())[0]   
                                                       + grad(tau12.Other())[1],   grad(tau12.Other())[0]   
                                                        + grad(tau2.Other())[1]) ) )                   

h = specialcf.mesh_size

# ********* bilinear forms ****** #

a = BilinearForm (fes, symmetric=True)
a +=  InnerProduct(divsigma, divtau)*dx + InnerProduct(Asig, tau)*dx 
a += sota*k**2/h*InnerProduct(jump_sigma, jump_tau) * dx(skeleton=True )
a += -(mean_divsigma*jump_tau + mean_divtau*jump_sigma) * dx(skeleton=True )
a += sota*k**2/h*(jump_sigma*jump_tau) * ds(skeleton=True, 
                                            definedon=mesh.Boundaries("left|right|top") )
a += -(mean_divsigma*jump_tau + mean_divtau*jump_sigma) * ds(skeleton=True 
                                                             , definedon=mesh.Boundaries("left|right|top") )

m = BilinearForm (fes, symmetric=True)
m += InnerProduct(Asig, tau)*dx 


a.Assemble()
m.Assemble()

# ********* solving the eigenproblem ****** #

u = GridFunction(fes, multidim=25, name='resonances')

with TaskManager():
    lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), list(u.vecs), shift= 0.01)


eigen = []
for evals in lam:
    if evals.real > 1.0000001: eigen.append(np.sqrt(evals.real - 1))

# ********* listing the eigenvalues ****** #

for i in range(len(eigen)):
        print("%1.6f"  %(eigen[i]))