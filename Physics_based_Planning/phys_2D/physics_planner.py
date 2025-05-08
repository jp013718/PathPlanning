"""
PHYSICS_BASED_PLANNING
@author: James Pennington
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Self

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Physics_based_Planning.phys_2D import env

class Node:
  def __init__(self, n: list[int]):
    self.x = n[0]
    self.z= n[1]

class PhysPlanner:
  def __init__(self, s_start, s_goal, delta_t, iter_max, eps=0.01, granularity=0.1, gravity=9.81):
    self.s_start = Node(s_start)
    self.s_goal = Node(s_goal)
    self.delta_t = delta_t
    self.iter_max = iter_max
    self.eps = eps

    self.granularity = granularity
    self.gravity = gravity

    self.env = env.Env()
    self.x_range = self.env.x_range
    self.y_range = self.env.y_range
    self.obs_circle = self.env.obs_circle
    self.obs_rectangle = self.env.obs_rectangle
    self.obs_boundary = self.env.obs_boundary

    self.surf, self.vec_field = self.make_surf_and_vec_field()

    self.path = []

  def plan(self, init_vel, d_theta):
    for i in range(int(360/d_theta)):
      path = [self.s_start]
      x = self.s_start.x
      dx = init_vel*np.cos(i*np.pi/180)
      z = self.s_start.z
      dz = init_vel*np.sin(i*np.pi/180)
      y = self.surf[int(self.s_start.x/self.granularity)][int(self.s_start.y/self.granularity)]
      dy = 0
      
      for _ in range(self.iter_max):
        col = int(x/self.granularity)
        row = int(z/self.granularity)

        dsdx = np.array([1, self.vec_field["x"][col][row], 0])
        dsdz = np.array([0, self.vec_field["z"][col][row], 1])

        norm = np.cross(dsdx, dsdz)
        unit_norm = norm/np.linalg.norm(norm)

        norm_accel = np.dot(unit_norm, np.array([0, self.gravity, 0]))
        ddx = norm_accel[0]
        ddy = norm_accel[1]-self.gravity
        ddz = norm_accel[2]

        dx += ddx
        dy += ddy
        dz += ddz
        x += dx
        y += dy
        z += dz

        path.append(Node([x, z]))
        
        if np.sqrt((self.s_goal.x - x)**2+(self.s_goal.z - z)**2) < self.eps:
          self.path = self.shortest_path(path, self.path)


  def make_surf_and_vec_field(self, obs_func):
    surf = np.zeros(shape=(self.x_range/self.granularity, self.y_range/self.granularity))
    # Add circular obstacles
    for obs in self.obs_circle:
      for row in surf:
        for square in row:
          pass
    
    # Add rectangular obstacles
    for obs in self.obs_rectangle:
      for row in surf:
        for square in row:
          pass

    # Add boundaries
    for obs in self.obs_boundary:
      for row in surf:
        for square in row:
          pass

    # Add goal
    for i in range(len(surf)):
      for j in range(len(surf[i])):
        z = i*self.granularity
        x = j*self.granularity

        surf[i][j] += 1/(np.sqrt((x-self.s_goal.x)**2+(z-self.s_goal.z)**2)**2) if x != self.s_goal.x and z != self.s_goal.z else -np.inf

    x_vec_field = np.zeros_like(surf)
    z_vec_field = np.zeros_like(surf)
    for i in range(len(surf)):
      for j in range(len(surf[i])):
        x_vec_field[i][j] = (surf[i-1][j]-surf[i+1][j])/2
        z_vec_field[i][j] = (surf[i][j-1]-surf[i][j+1])/2

    return surf, {'x': x_vec_field, 'z': z_vec_field}
  

  def shortest_path(self, path1: list[Node], path2: list[Node]):
    len_path1 = 0
    for i in range(len(path1)-1):
      len_path1 += np.sqrt((path1[i].x-path1[i+1].x)**2+(path1[i].z-path1[i+1].z)**2)
    
    len_path2 = 0
    for i in range(len(path2)-1):
      len_path2 += np.sqrt((path2[i].x-path2[i+1].x)**2+(path2[i].z-path2[i+1].z)**2)

    return path1 if len_path1 <= len_path2 else path2