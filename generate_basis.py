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
path = 'work/mesh3'
mesh = chio.read_mesh(f'{path}/model', meshio=True)
bdata = chio.read_bfile(f'{path}/model', element='tetra')
endo_patch = 19
epi_patch = 7
inflow_patchs = [6, 15]
outflow_patchs = [2, 10, 14]

# Boundary conditions
bc_dict = {}
for inflow_patch in inflow_patchs:
    bc_dict[inflow_patch] = 0.0  # Inflow boundary condition
for outflow_patch in outflow_patchs:
    bc_dict[outflow_patch] = 1.0  # Outflow boundary condition

long_bcs_marker = {'face': bc_dict}
trans_bcs_marker = {'face': {endo_patch: 0.0,
                             epi_patch: 1.0,}}

# Initialize Laplace problem
dxmesh, mt = dxio.read_meshio_mesh(mesh, bdata)
corr, _ = dxio.find_vtu_dx_mapping(dxmesh)
solver = LaplaceProblem(dxmesh, mt)

# Solve for the longitudinal and transmural basis functions
print('Solving for longitudinal basis functions...')
long, gL = solve_laplace(solver, long_bcs_marker, corr)
chio.write_dfile(f'{path}/longitudinal.FE', long)

print('Solving for transmural basis functions...')
trans, gT = solve_laplace(solver, trans_bcs_marker, corr)
chio.write_dfile(f'{path}/transmural.FE', trans)

# # Get circumferential coordinate
# print('Calculating circumferential coordinate...')
# center, long_values = ut.get_centerline(mesh, bdata, long, endo_patch, ndiv=21)
# node_center, node_normals = ut.assign_centerline_to_mesh(long, long_values, center)

# circ = ut.get_circumferential_coord(mesh, bdata, endo_patch, node_center, node_normals)
# chio.write_dfile(f'{path}/circumferential.FE', circ)

mesh.point_data['long'] = long
mesh.point_data['trans'] = trans
# mesh.point_data['circ'] = circ
io.write(f'{path}/basis.vtu', mesh)