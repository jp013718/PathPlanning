"""
Ray_Tracing_2D
@author: James Pennington
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import rc_context
from typing import Self

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + 
                "/../../")
from Sampling_based_Planning.rrt_2D import env, plotting, utils

# Cartesian point class
class Point:
  def __init__(self, n, pred=None):
    self.x = n[0]
    self.y = n[1]
    self.pred = pred
    self.next = None

  def __repr__(self):
    return f"({self.x}, {self.y})"

class Vector:
  def __init__(self, dx, dy):
    self.dx = dx
    self.dy = dy

  def __eq__(self, other: Self):
    return self.dx*other.dy-other.dx*self.dy == 0

# Line segment class
class Segment:
  def __init__(self, start:Point, end:Point):
    self.start = start
    self.end = end
    self.length = np.sqrt((end.x-start.x)**2+(end.y-start.y)**2)

  def colinear(self, p: Point):
    return self.end.x*p.y-self.end.y*p.x+self.start.y*p.x-self.start.x*p.y+self.start.x*self.end.y-self.start.y*self.end.x == 0

  def __repr__(self):
    return f"Start: ({self.start.x}, {self.start.y})\tEnd: ({self.end.x}, {self.end.y})"

# Rectangle class
class Rectangle:
  def __init__(self, origin:Point, width, height):
    self.width = width
    self.height = height
    
    self.corners = (
      origin, 
      Point((origin.x+width, origin.y)), 
      Point((origin.x, origin.y+height)), 
      Point((origin.x+width, origin.y+height))
    )
    
    self.sides = (
      Segment(self.corners[0], self.corners[1]), 
      Segment(self.corners[0], self.corners[2]), 
      Segment(self.corners[1], self.corners[3]), 
      Segment(self.corners[2], self.corners[3])
    )

# Circle class
class Circle:
  def __init__(self, center: Point, radius):
    self.center = center
    self.radius = radius

  def __repr__(self):
    return f"Centerpoint: ({self.center.x}, {self.center.y})\tRadius: {self.radius}"

# Path planning algorithm class
class Ray_Tracing:
  def __init__(self, s_start, s_goal, delta_theta, map_variant, theta_min=1, min_error=1e-3, threshold=1, random_cast=False):
    self.s_start = Point(s_start)
    self.s_goal = Point(s_goal)
    self.delta_theta = delta_theta
    self.theta_min = theta_min
    self.min_error = min_error
    self.threshold = threshold
    self.random_cast = random_cast
    self.path = [self.s_start]
    self.bad_points = []

    self.env = env.Env(map_variant)
    self.plotting = plotting.Plotting(s_start, s_goal)
    self.utils = utils.Utils()

    self.x_range = self.env.x_range
    self.y_range = self.env.y_range
    self.obs_circle = [Circle(Point((circ[0], circ[1])), circ[2]) for circ in self.env.obs_circle]
    self.obs_rectangle = [Rectangle(Point((rec[0], rec[1])), rec[2], rec[3]) for rec in self.env.obs_rectangle]
    self.obs_boundary = [Rectangle(Point((bound[0], bound[1])), bound[2], bound[3]) for bound in self.env.obs_boundary]

  # Function: plan
  # - Perform the main path-planning loop.
  # Output:
  # - path: List of points from s_start to s_goal following a legal path
  def plan(self):
    rays = self.cast_rays(delta_theta=self.delta_theta)
    
    # ########### DEBUG ############
    # fig, ax = self.visualize_env()
    # self.draw_rays(ax, rays)
    # self.draw_bad_points(ax)
    # self.draw_path(ax)
    # fig.show()
    # plt.show()
    # ###############################

    points = []
    # longest_ray = rays[0]
    # longest_ray_length = longest_ray.length
    for ray in rays:
      point = self.minimize_goal_distance(ray)
      # if longest_ray_length < ray.length:
      #   longest_ray = ray
      #   longest_ray_length = ray.length
      if all([self.dist(point[0], p) > self.threshold for p in self.bad_points]):
        points.append(point)

    """
    Adding the endpoint of the longest cast ray to points:
    - Allows solving when the distance to the goal is not strictly decreasing along
      a solution path
    - Breaks when backtracking is required
    """    
    # if all([self.dist(longest_ray.end, p) > self.threshold for p in self.bad_points]):
    #   points.append((longest_ray.end, self.goal_dist(longest_ray.end)))

    # Sort the points on the cast rays by their distance from the goal
    points = sorted(points, key=lambda pair: pair[1])
    while len(points) > 0:
      # Take the best point. If it reaches the goal, return the path...
      next_point = points.pop(0)
      if self.goal_dist(next_point[0]) < self.min_error:
        self.path.append(next_point[0])

        return self.path
      
      # If it doesn't reach the goal, make sure it's a sufficient distance from all other points on the path, 
      # then continue planning from that point
      elif all([self.dist(next_point[0], self.path[i]) > self.threshold for i in range(len(self.path))]):        
        theta = np.atan2(next_point[0].y-self.path[-1].y, next_point[0].x-self.path[-1].x)
        length = self.dist(self.path[-1], next_point[0])-0.1*self.min_error
        self.path.append(Point((length*np.cos(theta)+self.path[-1].x, length*np.sin(theta)+self.path[-1].y)))
        self.plan()
        # If the above planning step has found a viable path, return it. Otherwise, prune points that are near other
        # points that have not yielded results.
        if self.goal_dist(self.path[-1]) < self.min_error:
          return self.path
        else:
          self.bad_points.append(next_point[0])
          points = [point for point in points if all([self.dist(point[0], p) > self.threshold for p in self.bad_points])]

    self.path.pop()
          
  # Function: draw_path
  # - Draw the current path in a Matplotlib figure of the environment
  # Inputs:
  # - ax: The axes for the Matplotlib figure
  def draw_path(self, ax: plt.Axes):
    for i, p in enumerate(self.path[1:]):
        ax.plot([self.path[i].x, p.x], [self.path[i].y, p.y], "b-")

  # Function: cast_rays
  # - Cast rays radially about the last point on the path. If using random_cast, do so with random
  #   separation. Otherwise, uniformly cast rays with separation of delta_theta. Perform collision
  #   checking on cast rays to return the correctly terminated ray
  # Inputs:
  # - delta_theta: The desired angle separation of rays if not using random_cast
  # - radius: The distance rays should travel
  # Output:
  # - rays: List of segments with collision checking performed
  def cast_rays(self, delta_theta: float=1, radius=1000) -> tuple[Segment]:
    rays = []
    if self.random_cast:
      for _ in range(int(360//delta_theta)):
        rand_theta = 2*np.pi*np.random.random()
        ray = Segment(self.path[-1], Point((radius*np.cos(rand_theta)+self.path[-1].x, radius*np.sin(rand_theta)+self.path[-1].y)))
        rays.append(self.collision_check(ray))
    else:
      for i in range(int(360//delta_theta)):
        ray = Segment(self.path[-1], Point((radius*np.cos(i*delta_theta*np.pi/180)+self.path[-1].x, radius*np.sin(i*delta_theta*np.pi/180)+self.path[-1].y)))
        rays.append(self.collision_check(ray))
    
    return tuple(rays)

  # Function: draw_rays
  # - Draw cast rays onto a Matplotlib figure
  # Inputs:
  # - ax: The Matplotlib figure axes
  # - rays: A list of segments to draw
  def draw_rays(self, ax: plt.Axes, rays: list[Segment]):
    for ray in rays:
      ax.plot([ray.start.x, ray.end.x], [ray.start.y, ray.end.y], 'm-')

  def draw_bad_points(self, ax: plt.Axes):
    for point in self.bad_points:
      circle = plt.Circle((point.x, point.y), self.threshold, edgecolor="blue")
      ax.add_patch(circle)

  # Function: visualize_env
  # - Draw the environment (boundaries, obstacles, start, and goal) in a Matplotlib figure
  # Outputs:
  # - fig: The Matplotlib figure
  # - ax: The axes of the Matplotlib figure
  def visualize_env(self):
    fig, ax = plt.subplots()
    # Plot boundaries
    for bound in self.obs_boundary:
      rect = patches.Rectangle((bound.corners[0].x, bound.corners[0].y), bound.width, bound.height, facecolor="gray")
      ax.add_patch(rect)
    # Plot rectangle obstacles
    for obs in self.obs_rectangle:
      rect = patches.Rectangle((obs.corners[0].x, obs.corners[0].y), obs.width, obs.height, facecolor="gray")
      ax.add_patch(rect)
    # Plot circular obstacles
    for obs in self.obs_circle:
      circle = plt.Circle((obs.center.x, obs.center.y), obs.radius, facecolor="gray")
      ax.add_patch(circle)
    
    # Plot start and goal
    start = plt.Circle((self.s_start.x, self.s_start.y), 0.5, facecolor="green")
    goal = plt.Circle((self.s_goal.x, self.s_goal.y), 0.5, facecolor="red")
    ax.add_patch(start)
    ax.add_patch(goal)
    ax.axis("equal")

    return fig, ax

  # Function: minimize_point_distance
  # - Get the point on a projected ray closest to a given point
  # Inputs:
  # - ray: The projected ray
  # - point: The point to find the closest point to
  # Outputs:
  # - closest_point: The closest point on the ray to the specified point
  # - dist: The distance between the closest point and the specified point
  def minimize_point_distance(self, ray: Segment, point: Point) -> tuple[Point, float]:    
    # The point on a line closest to a point will always be at the intersection point of the line and
    # a perpendicular one passing through the point
    a = ray.start
    b = ray.end
    m = Vector(b.y-a.y, a.x-b.x)
    c = Segment(point, Point((point.x+m.dx, point.y+m.dy)))
    closest_point = self.check_intersection(ray, c, always_return_point=True)

    # Check that the calculated intersection is actually on the ray. If it is, return it. If not, return
    # whichever point is closer to the point between the start and end point of the ray.
    if np.abs(b.x-a.x) > 0.001*self.min_error:
      lambda_ = (closest_point.x-a.x)/(b.x-a.x)
    elif b.y-a.y != 0:
      lambda_ = (closest_point.y-a.y)/(b.y-a.y)
    else:
      print(ray)
      print(point)
      print(self.path)
      raise ValueError
    if 0 <= lambda_ <= 1:
      return (closest_point, self.dist(closest_point, point))
    else:
      closest_point = a if self.dist(a, point) < self.dist(b, point) else b
      return (closest_point, self.dist(closest_point, point))

  # Function: minimize_goal_distance
  # - Get the point on a projected ray closest to the goal
  # Input:
  # - ray: The projected ray
  # Outputs:
  # - closest_point: The closest point on the ray to the goal
  # - dist: The distance between the closest point and the goal
  def minimize_goal_distance(self, ray: Segment):
    return self.minimize_point_distance(ray, self.s_goal)

  # Function: dist
  # - Gets the euclidean distance between two points
  # Inputs:
  # - p1: First point
  # - p2: Second point
  # Output:
  # - Euclidean distance between p1 and p2
  def dist(self, p1: Point, p2: Point):
    return np.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)

  # Function: goal_dist
  # - Uses dist to get the distance between any point and the goal
  # Input:
  # - point: The point to get the goal distance from
  # Output:
  # - Distance from point to the goal
  def goal_dist(self, point: Point):
    return self.dist(point, self.s_goal)

  # Function: check_intersection
  # - Checks whether two segments intersect
  # Inputs:
  # - ab: First line segment
  # - cd: Second line segment
  # - always_return_point: If segments do not intersect, function will normally return None. Setting
  #   to True will return the projected intersection point.
  # Output:
  # - intersection: The intersection (or projected intersection) point of the two segments or None
  def check_intersection(self, ab: Segment, cd: Segment, always_return_point=False) -> Point|None:
    a = ab.start
    b = ab.end
    a_b = Vector(b.x-a.x, b.y-a.y)
    c = cd.start
    d = cd.end
    c_d = Vector(d.x-c.x, d.y-c.y)

    # Return None if the lines are parallel and nonintersecting
    if a_b == c_d and not ab.colinear(c) and not ab.colinear(d):
      return None
    
    # If the lines are colinear, return the point closer to ab.start
    if ab.colinear(c) and ab.colinear(d):
      return c if self.dist(a, c) < self.dist(a, d) else d

    a1 = a_b.dy
    b1 = -a_b.dx
    c1 = -a_b.dy*a.x+a_b.dx*a.y
    a2 = c_d.dy
    b2 = -c_d.dx
    c2 = -c_d.dy*c.x+c_d.dx*c.y

    # Solve for intersection point and return it if it is on both lines or if the point is requested
    intersection_x = (b1*c2-b2*c1)/(a1*b2-a2*b1)
    intersection_y = (c1*a2-c2*a1)/(a1*b2-a2*b1)
    intersection = Point((intersection_x, intersection_y))

    # Avoid errors when lines are near vertical
    if np.abs(a_b.dx) >= 0.001*self.min_error:
      lambda1 = (intersection.x-a.x)/(a_b.dx)
    else:
      lambda1 = (intersection.y-a.y)/(a_b.dy)

    if np.abs(c_d.dx) > 0.001*self.min_error:
      lambda2 = (intersection.x-c.x)/(c_d.dx)
    else:
      lambda2 = (intersection.y-c.y)/(c_d.dy)

    if 0 <= lambda1 <= 1 and 0 <= lambda2 <= 1 or always_return_point:
      return intersection
    else: 
      return None

  # Function: check_circle_intersection
  # - Checks whether a ray intersects a circle. If it does not, returns None. If it does, return
  #   the intersection point that is closer to the start point of the ray
  # Inputs:
  # - ray: The ray to check for intersection
  # - circle: The circle to check for intersection
  # Output:
  # - intersection: The intersection point that is closer to the start point of the ray if the ray
  #   and circle intersect. Otherwise, None.
  def check_circle_intersection(self, ray: Segment, circle: Circle) -> Point|None:
    x1 = ray.start.x - circle.center.x
    y1 = ray.start.y - circle.center.y
    x2 = ray.end.x - circle.center.x
    y2 = ray.end.y - circle.center.y
    r = circle.radius
    det = x1*y2-x2*y1
    dx = x2-x1
    dy = y2-y1
    dr = np.sqrt(dx**2+dy**2)
    sgn = lambda x: -1 if x < 0 else 1

    # Find the discriminant. If it's negative, the line and circle do not intersect
    discrim = (r**2)*(dr**2)-det**2
    if discrim < 0:
      return None
    
    x_int1 = (det*dy+sgn(dy)*dx*np.sqrt(discrim))/(dr**2)+circle.center.x
    y_int1 = (-det*dx+np.abs(dy)*np.sqrt(discrim))/(dr**2)+circle.center.y
    intersection1 = Point((x_int1, y_int1))

    x_int2 = (det*dy-sgn(dy)*dx*np.sqrt(discrim))/(dr**2)+circle.center.x
    y_int2 = (-det*dx-np.abs(dy)*np.sqrt(discrim))/(dr**2)+circle.center.y
    intersection2 = Point((x_int2, y_int2))

    # Make sure the intersection point is actually on the line segment
    intersection = intersection1 if self.dist(intersection1, ray.start) < self.dist(intersection2, ray.start) else intersection2
    if np.abs(dx) >= self.min_error:
      lambda_ = (intersection.x-ray.start.x)/dx
    else:
      lambda_ = (intersection.y-ray.start.y)/dy

    return intersection if 0 <= lambda_ <= 1 else None

    
  # Function: collision_check
  # - Checks whether a ray collides with any obstacles in the environment
  # Input:
  # - ray: The ray to check for collisions
  # Output:
  # - shortest_ray: The input ray shortened to the closest colliding object. Will return the
  #   original ray if no collisions occur
  def collision_check(self, ray: Segment) -> Segment:
    shortest_ray = ray
    # Check for collisions with environment boundaries
    for bound in self.obs_boundary:
      for seg in bound.sides:
        intersection = self.check_intersection(ray, seg)
        if intersection:
          if Segment(ray.start, intersection).length < shortest_ray.length:
            shortest_ray = Segment(ray.start, intersection)
            if shortest_ray.length == 0:
              print("Ray shortened to 0 in boundary check")
              print(f"Ray: {ray}")
              print(f"Boundary: {seg}")
              print(f"Intersection point: {intersection}")
              raise ValueError
    # Check for collisions with all rectangle obstacles in the environment
    for rec in self.obs_rectangle:
      for seg in rec.sides:
        intersection = self.check_intersection(ray, seg)
        if intersection:
          if Segment(ray.start, intersection).length < shortest_ray.length:
            shortest_ray = Segment(ray.start, intersection)
            if shortest_ray.length == 0 :
              print("Ray shortened to 0 in rectangle obstacle check")
    # Check for collisions with all circle obstacles in the environment
    for obstacle in self.obs_circle:
      intersection = self.check_circle_intersection(ray, obstacle)
      if intersection:
        if Segment(ray.start, intersection).length < shortest_ray.length:
          shortest_ray = Segment(ray.start, intersection)
          if shortest_ray.length == 0:
            print("Ray shortened to 0 in circular obstacle check")
    
    return shortest_ray
  
  # Function: angle_to_goal
  # - Gets angle from horiztonal to line between the most recent point on the path to the goal
  # Output:
  # - Angle to goal
  def angle_to_goal(self):
    return np.atan2(self.s_goal.y-self.path[-1].y, self.s_goal.x-self.path[-1].x)
  


if __name__=="__main__":
  import argparse
  from datetime import datetime as dt

  # Arguments: Start, Goal, Delta_Theta, Variant, Random_Cast
  parser = argparse.ArgumentParser()
  parser.add_argument("start_x", type=int)
  parser.add_argument("start_y", type=int)
  parser.add_argument("goal_x", type=int)
  parser.add_argument("goal_y", type=int)
  parser.add_argument("--d_theta", "-d", type=float, default=0.5)
  parser.add_argument("--map", "-m", type=int, default=0)
  parser.add_argument("--random_cast", "-r", action="store_true")
  parser.add_argument("--min_error", "-e", type=float, default=1e-3)
  parser.add_argument("--threshold", "-t", type=float, default=1)
  parser.add_argument("--visualize", "-v", action="store_true")

  args = parser.parse_args()

  ray_trace = Ray_Tracing(s_start=(args.start_x, args.start_y), s_goal=(args.goal_x, args.goal_y), delta_theta=args.d_theta, map_variant=args.map, theta_min=1, min_error=args.min_error, threshold=args.threshold, random_cast=args.random_cast)
  try:
    start_time = dt.now()
    path = ray_trace.plan()
    finish_time = dt.now()
    print(f"Elapsed Time: {finish_time-start_time}")
    print(f"Path: {path}")
    if args.visualize:
      fig1, ax1 = ray_trace.visualize_env()
      ray_trace.draw_path(ax1)
      fig1.show()
      
      fig2, ax2 = ray_trace.visualize_env()
      ray_trace.draw_bad_points(ax2)
      fig2.show()
      plt.show()
  except KeyboardInterrupt as e:
    print(ray_trace.path)
    fig, ax = ray_trace.visualize_env()
    ray_trace.draw_path(ax)
    fig.show()
    fig, ax = ray_trace.visualize_env()
    ray_trace.draw_bad_points(ax)
    fig.show()
    plt.show()
