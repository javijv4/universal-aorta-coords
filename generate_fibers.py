#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  17 07:42:57 2023

@author: Javiera Jilberto Vallejos
"""

import numpy as np
import meshio as io

alpha = 45
basis = io.read('basis.vtu')

gT = basis.point_data['gT'] 
gL = basis.point_data['gLong'] 
gC = basis.point_data['gC'] 
trans = basis.point_data['trans']

fib1 = gC*np.cos(alpha) + np.cross(gT, gC, axisa=1, axisb=1)*np.sin(alpha) + gT*np.sum(gT*gC, axis=1)[:,None]*(1-np.cos(alpha))
fib2 = gC*np.cos(-alpha) + np.cross(gT, gC, axisa=1, axisb=1)*np.sin(-alpha) + gT*np.sum(gT*gC, axis=1)[:,None]*(1-np.cos(-alpha))

mesh = io.Mesh(basis.points, basis.cells)
mesh.point_data['fib1'] = fib1
mesh.point_data['fib2'] = fib2
mesh.point_data['trans'] = trans

io.write('fibers.vtu', mesh)