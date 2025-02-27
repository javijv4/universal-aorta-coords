#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Created on 2025/02/27 08:56:55

@author: Javiera Jilberto Vallejos 
'''

import meshio as io
import pyvista as pv
import numpy as np
from scipy.interpolate import interp1d


def get_normal_plane_svd(points):   # Find the plane that minimizes the distance given N points
    centroid = np.mean(points, axis=0)
    svd = np.linalg.svd(points - centroid)
    normal = svd[2][-1]
    normal = normal/np.linalg.norm(normal)
    return normal, centroid


def get_centerline(mesh, bdata, long, endo_patch, ndiv=21):
    # Get endo surface
    tri_endo = bdata[bdata[:,-1] == endo_patch, 1:-1]

    # Generate surface mesh
    tri_mesh = io.Mesh(points=mesh.points, cells=[('triangle', tri_endo)], point_data={'lap': long})
    mesh = pv.from_meshio(tri_mesh)

    # Find contours along the centerline
    div_values = np.linspace(0,1,ndiv)

    contours = []
    centers = []
    values = []
    for val in div_values:
        contour = mesh.contour(isosurfaces=[val])

        points = contour.points

        contours.append(points)
        if len(points) == 0:
            continue

        values.append(val)

        center = np.mean(points, axis=0)
        centers.append(center)

    centers = np.vstack(centers)

    return centers, values


def assign_centerline_to_mesh(long, values, centers):
    # interpolate centerline and evaluate at each node
    center_func = interp1d(values, centers.T, kind='cubic', fill_value='extrapolate')   
    node_center = np.array(center_func(long)).T

    # Find normals and evaluate at each node
    sampled_lap = np.linspace(0,1,100)
    sampled_centers = center_func(sampled_lap).T
    sampled_normals = np.diff(sampled_centers, axis=0)
    sampled_normals = np.vstack([sampled_normals, sampled_normals[-1]])

    normal_func = interp1d(sampled_lap, sampled_normals.T, kind='cubic', fill_value='extrapolate')
    node_normals = np.array(normal_func(long)).T
    node_normals = node_normals/np.linalg.norm(node_normals, axis=1)[:,None]

    return node_center, node_normals


def get_circumferential_coord(mesh, bdata, endo_patch, node_center, node_normals):
    # Find centroid and aorta plane
    centroid = np.mean(np.vstack(node_center), axis=0)

    # Grab surface nodes to calculate the aorta plane (using all the nodes it's too much memory)
    surface_points = mesh.points[np.unique(bdata[bdata[:,-1] == endo_patch, 1:-1])]

    aorta_plane, _ = get_normal_plane_svd(surface_points)
    aorta_plane = aorta_plane/np.linalg.norm(aorta_plane)

    # Find the vector that goes from the center to the centroid
    aux = node_center - centroid
    aux = aux/np.linalg.norm(aux, axis=0)

    # Calculate out of plane vector
    ex = np.cross(aux, node_normals, axisa=1, axisb=1)
    ex = ex/np.linalg.norm(ex, axis=0)

    node_ex = np.tile(aorta_plane, (node_normals.shape[0], 1))


    # Calculate ey
    ey = np.cross(node_ex, node_normals, axisa=1, axisb=1)
    node_ey = ey/np.linalg.norm(ey, axis=1)[:,None]
    node_vector = mesh.points - node_center

    node_vx = np.sum(node_vector*node_ex, axis=1)
    node_vy = np.sum(node_vector*node_ey, axis=1)
    node_theta = np.arctan2(node_vx, node_vy)

    node_theta = node_theta/np.pi

    return node_theta