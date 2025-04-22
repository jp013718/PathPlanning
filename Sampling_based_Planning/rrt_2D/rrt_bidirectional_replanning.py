"""
DYNAMIC_RRT_2D_BIDIRECTIONAL_REPLANNING
@author: James Pennington, Kaleb Keichel
@credit: huiming zhou
"""

import os
import sys
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from typing import Self

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../")

from Sampling_based_Planning.rrt_2D import env, plotting, utils


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        self.flag = "VALID"

    def __repr__(self):
        return f"Node: ({self.x}, {self.y})"


class Edge:
    def __init__(self, n_p, n_c):
        self.parent = n_p
        self.child = n_c
        self.flag = "VALID"

    def __repr__(self):
        return f"EDGE: Child - {self.child}\t|\tParent - {self.parent}"
    
    def intersect(self, other: Self):
        a1 = self.parent.y - self.child.y
        b1 = -(self.parent.x - self.child.x)
        c1 = -a1*self.child.x - b1*self.child.y
        a2 = other.parent.y - other.child.y
        b2 = -(other.parent.x - other.child.x)
        c2 = -a2*other.child.x - b2*other.child.y

        if a1*b2-a2*b1 == 0:
            print(f"{a1}, {b1}, {c1}, {a1}, {b2}, {c2}")
            print(self)
            print(other)

        if self.child.x*other.parent.y - self.child.y*other.parent.x == 0:
            return None

        intersection = Node(((b1*c2-b2*c1)/(a1*b2-a2*b1), (c1*a2-c2*a1)/(a1*b2-a2*b1)))

        if np.abs(a1) > np.abs(b1):
            lambda1 = (intersection.y-self.child.y)/(self.parent.y-self.child.y)
        else:
            lambda1 = (intersection.x-self.child.x)/(self.parent.x-self.child.x)
        
        if np.abs(a1) > np.abs(b2):
            lambda2 = (intersection.y-other.child.y)/(other.parent.y-other.child.y)
        else:
            lambda2 = (intersection.x-other.child.x)/(other.parent.x-other.child.x)

        return intersection if 0 <= lambda1 <= 1 and 0 <= lambda2 <= 1 else None
    
    def is_equivalent(self, other: Self):
        return self.child.x == other.child.x and self.child.y == other.child.y and self.parent.x == other.parent.x and self.parent.y == other.parent.y
        


class DrrtConnect:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, waypoint_sample_rate, iter_max):
        self.s_start = Node(s_start)
        self.s_goal = Node(s_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.waypoint_sample_rate = waypoint_sample_rate
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.vertex_old = []
        self.vertex_new = []
        self.edges = []
        # Idea: Maintain individual trees in this list. We should only need pointers
        # to the root nodes
        self.roots = [self.s_start]
        self.trees = []
        self.tree_edges = []
        self.goal_node = None

        self.env = env.Env()
        self.plotting = plotting.Plotting(s_start, s_goal)
        self.utils = utils.Utils()
        self.fig, self.ax = plt.subplots()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        self.obs_add = [0, 0, 0]

        self.path = []
        self.waypoint = []

    def planning(self):
        for i in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if self.get_distance_and_angle(node_new, node_near)[0] == 0:
                continue

            if node_new and not self.utils.is_collision(node_near, node_new):
                self.vertex.append(node_new)
                self.edges.append(Edge(node_near, node_new))
                dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                if dist <= self.step_len:
                    self.new_state(node_new, self.s_goal)

                    path = self.extract_path(node_new)
                    self.goal_node = node_new
                    self.plot_grid("Dynamic_RRT")
                    self.plot_visited()
                    self.plot_path(path)
                    self.path = path
                    self.waypoint = self.extract_waypoint(node_new)
                    self.fig.canvas.mpl_connect('button_press_event', self.on_press)
                    plt.show()

                    return

        return None

    def on_press(self, event):
        x, y = event.xdata, event.ydata
        if x < 0 or x > 50 or y < 0 or y > 30:
            print("Please choose right area!")
        else:
            x, y = int(x), int(y)
            print("Add circle obstacle at: s =", x, ",", "y =", y)
            self.obs_add = [x, y, 2]
            self.obs_circle.append([x, y, 2])
            self.utils.update_obs(self.obs_circle, self.obs_boundary, self.obs_rectangle)
            self.InvalidateNodes()

            if self.is_path_invalid():
                print("Path is Replanning ...")
                path, waypoint = self.replanning()

                print("len_vertex: ", len(self.vertex))
                print("len_vertex_old: ", len(self.vertex_old))
                print("len_vertex_new: ", len(self.vertex_new))

                plt.cla()
                self.plot_grid("Dynamic_RRT")
                self.plot_vertex_old()
                self.plot_path(self.path, color='blue')
                self.plot_vertex_new()
                self.vertex_new = []
                self.plot_path(path)
                self.path = path
                self.waypoint = waypoint
            else:
                print("Trimming Invalid Nodes ...")
                self.TrimRRT()

                plt.cla()
                self.plot_grid("Dynamic_RRT")
                self.plot_visited(animation=False)
                self.plot_path(self.path)

            self.fig.canvas.draw_idle()

    #######################################################
    #
    # TODO: Invalidate nodes using windowing queries
    # - May require significant changes to added obstacles
    #   as those are currently always circles with radius 2
    #
    #######################################################
    def InvalidateNodes(self):
        for edge in self.edges:
            if self.is_collision_obs_add(edge.parent, edge.child):
                edge.child.flag = "INVALID"

    def is_path_invalid(self):
        for node in self.waypoint:
            if node.flag == "INVALID":
                return True

    # Check if a given edge (start - end) intersects the most recently added obstacle. This obstacle will always
    # be a circle centered at (x, y) with a radius of 2
    def is_collision_obs_add(self, start, end):
        delta = self.utils.delta
        obs_add = self.obs_add

        # Check if start point is inside the most recently added obstacle
        if math.hypot(start.x - obs_add[0], start.y - obs_add[1]) <= obs_add[2] + delta:
            return True

        # Check if end point is inside the most recently added obstacle
        if math.hypot(end.x - obs_add[0], end.y - obs_add[1]) <= obs_add[2] + delta:
            return True

        # Check if the edge between start and end points intersects the most recently added obstacle
        o, d = self.utils.get_ray(start, end)
        if self.utils.is_intersect_circle(o, d, [obs_add[0], obs_add[1]], obs_add[2]):
            return True

        return False

    ########################################################
    #
    # TODO: Merge trees when an intersection occurs
    # - Need to figure out how to rebuild tree structure
    #   when merge occurs
    # - Also involves reassigning parents when merges happen
    #
    ########################################################
    def replanning(self):
        self.TrimRRT()

        for i in range(self.iter_max):
            print(f"i: {i}")
            j = 0
            if len(self.roots) == 1:
                print(f"Start: {self.s_start}")
                goal_root = self.goal_node
                while goal_root.parent:
                    print(f"On path: {goal_root}")
                    goal_root = goal_root.parent
                print(f"Goal Root: {goal_root}")
                print(f"Remaining root: {self.roots[0]}")
                break
            while j < len(self.roots):
                print(f"\tj: {j}")
                print(f"Number of trees: {len(self.roots)}")
                tree = self.trees[j]
                node_rand = self.generate_random_node_replanning(self.goal_sample_rate, self.waypoint_sample_rate)
                node_near = self.nearest_neighbor(tree, node_rand)
                node_new = self.new_state(node_near, node_rand)

                if self.get_distance_and_angle(node_new, node_near)[0] == 0:
                    j += 1
                    continue

                if node_new and not self.utils.is_collision(node_near, node_new):
                    self.vertex.append(node_new)
                    self.vertex_new.append(node_new)
                    tree.append(node_new)
                    edge_new = Edge(node_near, node_new)
                    intersection = None
                    intersecting_edge = None
                    for k, edge_tree in enumerate(self.tree_edges):
                        if k == j: continue
                        for edge in edge_tree:
                            if edge.intersect(edge_new):
                                intersection = edge.intersect(edge_new)
                                i_tree_index = k
                                intersecting_edge = edge


                    if intersection:
                        edge_child, edge_parent = intersecting_edge.child, intersecting_edge.parent
                        self.edges = [edge for edge in self.edges if not edge.is_equivalent(intersecting_edge)]
                        self.tree_edges[i_tree_index] = [edge for edge in self.tree_edges[i_tree_index] if not edge.is_equivalent(intersecting_edge)]
                        self.vertex.append(intersection)
                        tree.append(intersection)
                        # MERGE TREES
                        # Break the intersecting edges...
                        # Add a new edge from the child of the intersecting edge to the intersection point
                        self.edges.append(Edge(intersection, edge_child))
                        self.tree_edges[i_tree_index].append(Edge(intersection, edge_child))
                        edge_child.parent = intersection
                        # Add a new edge from the new point to the intersection point
                        self.edges.append(Edge(intersection, node_new))
                        self.tree_edges[j].append(Edge(intersection, node_new))
                        node_new.parent = intersection
                        
                        # If the parent node of the intersecting edge is part of the start tree,
                        # make sure it doesn't flip the start node
                        if self.in_tree(edge_parent, self.s_start):
                            print("Flipping when intersecting tree contains start")
                            self.edges.append(Edge(edge_parent, intersection))
                            self.tree_edges[i_tree_index].append(Edge(edge_parent, intersection))
                            intersection.parent = edge_parent
                            self.flip_tree(node_near, intersection, j)
                            for node in tree:
                                self.trees[i_tree_index].append(node)
                            for edge in self.tree_edges[j]:
                                self.tree_edges[i_tree_index].append(edge)
                            self.trees.pop(j)
                            self.roots.pop(j)
                            self.tree_edges.pop(j)
                        # Otherwise, we flip the tree containing the intersecting edge
                        else:
                            print("Flipping when intersecting tree does not contain start")
                            self.edges.append(Edge(node_near, intersection))
                            self.tree_edges[j].append(Edge(node_near, intersection))
                            intersection.parent = node_near
                            self.flip_tree(edge_parent, intersection, i_tree_index)
                            for node in self.trees[i_tree_index]:
                                tree.append(node)
                            for edge in self.tree_edges[i_tree_index]:
                                self.tree_edges[j].append(edge)
                            self.trees.pop(i_tree_index)
                            self.roots.pop(i_tree_index)
                            self.tree_edges.pop(i_tree_index)

                        if self.in_tree(self.goal_node, self.s_start):
                            path = self.extract_path(self.goal_node)
                            waypoint = self.extract_waypoint(self.goal_node)
                            print("path: ", len(path))
                            print("waypoint: ", len(waypoint))

                            return path, waypoint
                    else:
                        self.edges.append(edge_new)
                        self.tree_edges[j].append(edge_new)
                        j += 1

                j += 1
        
        return None
    
    def flip_tree(self, node: Node, new_root: Node, index):
        prev = new_root
        cursor = node
        next = node.parent
        while cursor.parent:
            # Since edges currently aren't a sorted list, it's probably just as efficient to rebuild the
            # list in order to delete an edge from it
            self.edges = [edge for edge in self.edges if not edge.is_equivalent(Edge(next, cursor))]
            self.tree_edges[index] = [edge for edge in self.tree_edges[index] if not edge.is_equivalent(Edge(next, cursor))]
            cursor.parent = prev
            self.edges.append(Edge(cursor.parent, cursor))
            self.tree_edges[index].append(Edge(cursor.parent, cursor))
            prev = cursor
            cursor = next
            next = cursor.parent

    #####################################################################
    #
    # TODO: Improvements...
    # - Multiple list comprehension methods are likely less efficient
    #   then using a loop to build all these lists at once
    # - Setting parent nodes to None may be doable at the same time we 
    #   invalidate nodes instead of in a separate loop here
    #
    #####################################################################
    def TrimRRT(self):
        # Node invalidation occurs in the obstacle addition event, so all necessary nodes should be 
        # invalidated by this point
        
        # For all nodes in the current tree, if the parent node is invalid, dereference the parent node
        for i in range(1, len(self.vertex)):
            node = self.vertex[i]
            node_p = node.parent
            if not node_p: continue
            if node_p.flag == "INVALID":
                node.parent = None

        # Remove invalid nodes
        self.vertex = [node for node in self.vertex if node.flag == "VALID"]
        self.vertex_old = copy.deepcopy(self.vertex)
        # Get the root node of each remaining tree
        self.roots = [node for node in self.vertex if node.parent is None]
        # Maintain nodes in their individual trees
        self.trees = [[node for node in self.vertex if self.in_tree(node, tree)] for tree in self.roots]
        self.edges = [Edge(node.parent, node) for node in self.vertex[1:len(self.vertex)] if node.parent]
        self.tree_edges = [[edge for edge in self.edges if self.in_tree(edge.child, tree)] for tree in self.roots]

        
    # Backtrack up through the parents of node until we reach the root, then compare roots
    def in_tree(self, node: Node, root: Node):
        new_node = node.parent if node.parent else node
        while new_node.parent:
            new_node = new_node.parent
        return new_node is root

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def generate_random_node_replanning(self, goal_sample_rate, waypoint_sample_rate):
        delta = self.utils.delta
        p = np.random.random()

        if p < goal_sample_rate:
            return self.s_goal
        elif goal_sample_rate < p < goal_sample_rate + waypoint_sample_rate:
            return self.waypoint[np.random.randint(0, len(self.waypoint) - 1)]
        else:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    def extract_waypoint(self, node_end):
        waypoint = [self.s_goal]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            waypoint.append(node_now)

        return waypoint

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)

    def plot_grid(self, name):

        for (ox, oy, w, h) in self.obs_boundary:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )

        for (ox, oy, w, h) in self.obs_rectangle:
            self.ax.add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        for (ox, oy, r) in self.obs_circle:
            self.ax.add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='gray',
                    fill=True
                )
            )

        plt.plot(self.s_start.x, self.s_start.y, "bs", linewidth=3)
        plt.plot(self.s_goal.x, self.s_goal.y, "gs", linewidth=3)

        plt.title(name)
        plt.axis("equal")

    def plot_visited(self, animation=True):
        if animation:
            count = 0
            for node in self.vertex:
                count += 1
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")
                    plt.gcf().canvas.mpl_connect('key_release_event',
                                                 lambda event:
                                                 [exit(0) if event.key == 'escape' else None])
                    if count % 10 == 0:
                        plt.pause(0.001)
        else:
            for node in self.vertex:
                if node.parent:
                    plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    def plot_vertex_old(self):
        for node in self.vertex_old:
            if node.parent:
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], "-g")

    def plot_vertex_new(self):
        count = 0

        for node in self.vertex_new:
            count += 1
            if node.parent:
                plt.plot([node.parent.x, node.x], [node.parent.y, node.y], color='darkorange')
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event:
                                             [exit(0) if event.key == 'escape' else None])
                if count % 10 == 0:
                    plt.pause(0.001)

    @staticmethod
    def plot_path(path, color='red'):
        plt.plot([x[0] for x in path], [x[1] for x in path], linewidth=2, color=color)
        plt.pause(0.01)


def main():
    x_start = (2, 2)  # Starting node
    x_goal = (49, 24)  # Goal node

    drrt = DrrtConnect(x_start, x_goal, 0.5, 0.1, 0.6, 50000)
    drrt.planning()


if __name__ == '__main__':
    main()
