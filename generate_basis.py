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
mesh = chio.read_mesh('mesh/model', meshio=True)
bdata = chio.read_bfile('mesh/model', element='tetra')
endo_patch = 3

# Boundary conditions
long_bcs_marker = {'face': {1: 0.0, 12: 0.0,
                            5: 1.0, 8: 1.0}}
trans_bcs_marker = {'face': {3: 0.0,
                        14: 1.0,}}

# Initialize Laplace problem
dxmesh, mt = dxio.read_meshio_mesh(mesh, bdata)
corr, _ = dxio.find_vtu_dx_mapping(dxmesh)
solver = LaplaceProblem(dxmesh, mt)

# Solve for the longitudinal and transmural basis functions
long, gL = solve_laplace(solver, long_bcs_marker, corr)
trans, gT = solve_laplace(solver, trans_bcs_marker, corr)

# Get circumferential coordinate
center, long_values = ut.get_centerline(mesh, bdata, long, endo_patch, ndiv=21)
node_center, node_normals = ut.assign_centerline_to_mesh(long, long_values, center)

circ = ut.get_circumferential_coord(mesh, bdata, endo_patch, node_center, node_normals)

# Save basis functions
chio.write_dfile('longitudinal.FE', long)
chio.write_dfile('transmural.FE', trans)
chio.write_dfile('circumferential.FE', circ)

mesh.point_data['long'] = long
mesh.point_data['trans'] = trans
mesh.point_data['circ'] = circ
io.write('basis.vtu', mesh)