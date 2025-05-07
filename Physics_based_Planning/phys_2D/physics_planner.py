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

class node:
  def __init__(self, n: list[int]):
    self.x = n[0]
    self.y= n[1]

class PhysPlanner:
  def __init__(self, s_start, s_goal, delta_t, iter_max, granularity = 0.1, gravity = 9.81):
    self.s_start = node(s_start)
    self.s_goal = node(s_goal)
    self.delta_t = delta_t
    self.iter_max = iter_max

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
      y = self.s_start.y
      dy = init_vel*np.sin(i*np.pi/180)
      z = self.surf[self.s_start.x/self.granularity][self.s_start.y/self.granularity]
      dz = 0
      
      for j in range(self.iter_max):
        pass

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
        y = i*self.granularity
        x = j*self.granularity

        surf[i][j] += 1/(np.sqrt((x-self.s_goal.x)**2+(y-self.s_goal.y)**2)**2) if x != self.s_goal.x and y != self.s_goal.y else -np.inf

    x_vec_field = np.zeros_like(surf)
    y_vec_field = np.zeros_like(surf)
    for i in range(len(surf)):
      for j in range(len(surf[i])):
        x_vec_field[i][j] = (surf[i-1][j]-surf[i+1][j])/2
        y_vec_field[i][j] = (surf[i][j-1]-surf[i][j+1])/2

    return surf, {'x': x_vec_field, 'y': y_vec_field}