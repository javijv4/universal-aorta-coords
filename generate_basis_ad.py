#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  17 07:42:57 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import cheartio as chio
import meshio as io
from LaplaceProblem import LaplaceProblem, solve_laplace
import dolfinxio as dxio
import utils as ut

# Read-in mesh and boundary file
mesh = chio.read_mesh('work/solidonly', meshio=True)
bdata = chio.read_bfile('work/solidonly', element='tetra')
endo_patch = 19

alpha = 45

# Boundary conditions
long_bcs_marker = {'face': {15: 0.0,
                            2: 1.0}}
trans_bcs_marker = {'face': {19: 0.0,
                        7: 1.0,}}
trans_flap_bcs_marker = {'face': {17: 0.0,
                        18: 1.0,}}


# Initialize Laplace problem
dxmesh, mt = dxio.read_meshio_mesh(mesh, bdata)
corr, _ = dxio.find_vtu_dx_mapping(dxmesh)
solver = LaplaceProblem(dxmesh, mt)

# Solve for the longitudinal and transmural basis functions
long, gL = solve_laplace(solver, long_bcs_marker, corr)
trans, gT = solve_laplace(solver, trans_bcs_marker, corr)
trans_flap, gTf = solve_laplace(solver, trans_flap_bcs_marker, corr)

# Get circumferential coordinate
center, long_values = ut.get_centerline(mesh, bdata, long, endo_patch, ndiv=21)
node_center, node_normals = ut.assign_centerline_to_mesh(long, long_values, center)

circ = ut.get_circumferential_coord(mesh, bdata, endo_patch, node_center, node_normals)

# Get circumferential vector
gC = np.cross(gL, gT, axisa=1, axisb=1)
gC = gC / np.linalg.norm(gC, axis=1)[:, None]

gCf = np.cross(gL, gTf, axisa=1, axisb=1)
gCf = gCf / np.linalg.norm(gCf, axis=1)[:, None]

# Save basis functions
chio.write_dfile('longitudinal.FE', long)
chio.write_dfile('transmural.FE', trans)
chio.write_dfile('transmural_flap.FE', trans)
chio.write_dfile('circumferential.FE', circ)
chio.write_dfile('circumferential_flap.FE', circ)


# Generate fibers
fib1 = gC*np.cos(alpha) + np.cross(gT, gC, axisa=1, axisb=1)*np.sin(alpha) + gT*np.sum(gT*gC, axis=1)[:,None]*(1-np.cos(alpha))
fib2 = gC*np.cos(-alpha) + np.cross(gT, gC, axisa=1, axisb=1)*np.sin(-alpha) + gT*np.sum(gT*gC, axis=1)[:,None]*(1-np.cos(-alpha))

fib_flap1 = gCf*np.cos(alpha) + np.cross(gTf, gCf, axisa=1, axisb=1)*np.sin(alpha) + gTf*np.sum(gTf*gCf, axis=1)[:,None]*(1-np.cos(alpha))
fib_flap2 = gCf*np.cos(-alpha) + np.cross(gTf, gCf, axisa=1, axisb=1)*np.sin(-alpha) + gTf*np.sum(gTf*gCf, axis=1)[:,None]*(1-np.cos(-alpha))

# Distance field from flap
xyz = mesh.points
ien = mesh.cells[0].data
propagation = 10
flap_elems = bdata[(bdata[:,-1]==17)+(bdata[:,-1]==18), 1:-1]
flap_nodes = np.unique(flap_elems.ravel())

nx = np.zeros(len(xyz))
for i in range(len(ien)):
    nx[ien[i]] += 4

# Valve weight
flap_dist = np.zeros(len(xyz))
flap_dist[flap_nodes] = 1

# Propagation
for i in range(propagation):
    aux = flap_dist.copy()
    for j in range(len(ien)):
        ax = np.sum(flap_dist[ien[j]])
        aux[ien[j]] += ax
    aux = aux / nx
    aux[flap_nodes] = 1
    flap_dist = aux

# Cutoff
flap_dist[flap_dist > 1] = 1

# Combine both fields
fib1 = fib1 * (1-flap_dist[:, None]) + fib_flap1 * flap_dist[:, None]
fib2 = fib2 * (1-flap_dist[:, None]) + fib_flap2 * flap_dist[:, None]

fib1 = fib1 / np.linalg.norm(fib1, axis=1)[:, None]
fib2 = fib2 / np.linalg.norm(fib2, axis=1)[:, None]

mesh.point_data['long'] = long
mesh.point_data['trans'] = trans
mesh.point_data['circ'] = circ
mesh.point_data['gL'] = gL
mesh.point_data['gT'] = gT
mesh.point_data['gC'] = gC
mesh.point_data['gTf'] = gTf
mesh.point_data['gCf'] = gCf
mesh.point_data['fib1'] = fib1
mesh.point_data['fib2'] = fib2
mesh.point_data['fib_flap1'] = fib_flap1
mesh.point_data['fib_flap2'] = fib_flap2
mesh.point_data['flap_dist'] = flap_dist
io.write('basis.vtu', mesh)


# Save fibers
fib3 = np.cross(fib1, fib2, axisa=1, axisb=1)
fib3 = fib3 / np.linalg.norm(fib3, axis=1)[:, None]
save = np.column_stack((fib1, fib2, fib3))
chio.write_dfile('work/fibers.field', save)

io.write('fibers.vtu', mesh)