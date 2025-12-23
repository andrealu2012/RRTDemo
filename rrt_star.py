"""
RRT* 算法模块
"""

import numpy as np
from rrt_base import Node, RRT


class RRTStar(RRT):
    """RRT*算法实现 - RRT的优化版本"""

    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, search_radius=1.5):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter)
        self.search_radius = search_radius

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def find_near_nodes(self, new_node):
        near_nodes = []
        for node in self.nodes:
            if self.distance(node, new_node) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def choose_parent(self, new_node, near_nodes):
        if not near_nodes:
            return None
        min_cost = float('inf')
        best_parent = None
        for near_node in near_nodes:
            cost = near_node.cost + self.distance(near_node, new_node)
            if cost < min_cost and self.is_collision_free(near_node, new_node.x, new_node.y):
                min_cost = cost
                best_parent = near_node
        if best_parent:
            new_node.parent = best_parent
            new_node.cost = min_cost
        return best_parent

    def rewire(self, new_node, near_nodes):
        for near_node in near_nodes:
            new_cost = new_node.cost + self.distance(new_node, near_node)
            if new_cost < near_node.cost and self.is_collision_free(new_node, near_node.x, near_node.y):
                near_node.parent = new_node
                near_node.cost = new_cost
                self.update_children_cost(near_node)

    def update_children_cost(self, parent_node):
        for node in self.nodes:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self.distance(parent_node, node)
                self.update_children_cost(node)

    def plan(self):
        print("开始RRT*路径规划...")
        for i in range(self.max_iter):
            rand_x, rand_y = self.random_sample()
            nearest = self.nearest_node(rand_x, rand_y)
            new_x, new_y = self.steer(nearest, rand_x, rand_y)
            if not self.is_collision_free(nearest, new_x, new_y):
                continue
            new_node = Node(new_x, new_y)
            near_nodes = self.find_near_nodes(new_node)
            best_parent = self.choose_parent(new_node, near_nodes)
            if best_parent is None:
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest, new_node)
            self.nodes.append(new_node)
            self.rewire(new_node, near_nodes)
            if self.is_goal_reached(new_node):
                print(f"找到路径！迭代次数: {i+1}")
                self.goal.parent = new_node
                self.goal.cost = new_node.cost + self.distance(new_node, self.goal)
                return self.extract_path()
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.max_iter}...")
        print("未找到路径，达到最大迭代次数")
        return None
