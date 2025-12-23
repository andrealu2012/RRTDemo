"""
RRT*-Connect算法模块
"""

import numpy as np
from rrt_base import Node
from rrt_connect import RRTConnect


class RRTStarConnect(RRTConnect):
    """RRT*-Connect算法实现 - 结合双向搜索和路径优化"""

    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, search_radius=1.5):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter)
        self.search_radius = search_radius

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def find_near_nodes_in_tree(self, tree, new_node):
        near_nodes = []
        for node in tree:
            if self.distance(node, new_node) <= self.search_radius:
                near_nodes.append(node)
        return near_nodes

    def choose_parent_in_tree(self, tree, new_node, near_nodes):
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

    def rewire_tree(self, tree, new_node, near_nodes):
        for near_node in near_nodes:
            new_cost = new_node.cost + self.distance(new_node, near_node)
            if new_cost < near_node.cost and self.is_collision_free(new_node, near_node.x, near_node.y):
                near_node.parent = new_node
                near_node.cost = new_cost
                self.update_children_cost_in_tree(tree, near_node)

    def update_children_cost_in_tree(self, tree, parent_node):
        for node in tree:
            if node.parent == parent_node:
                node.cost = parent_node.cost + self.distance(parent_node, node)
                self.update_children_cost_in_tree(tree, node)

    def extend_tree_with_optimization(self, tree, rand_x, rand_y):
        nearest = self.nearest_node_in_tree(tree, rand_x, rand_y)
        new_x, new_y = self.steer(nearest, rand_x, rand_y)
        if not self.is_collision_free(nearest, new_x, new_y):
            return None
        new_node = Node(new_x, new_y)
        near_nodes = self.find_near_nodes_in_tree(tree, new_node)
        best_parent = self.choose_parent_in_tree(tree, new_node, near_nodes)
        if best_parent is None:
            new_node.parent = nearest
            new_node.cost = nearest.cost + self.distance(nearest, new_node)
        tree.append(new_node)
        self.rewire_tree(tree, new_node, near_nodes)
        return new_node

    def connect_tree_with_optimization(self, tree, target_x, target_y):
        max_extend_steps = 50
        step_count = 0
        while step_count < max_extend_steps:
            step_count += 1
            nearest = self.nearest_node_in_tree(tree, target_x, target_y)
            dx = target_x - nearest.x
            dy = target_y - nearest.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.step_size:
                if self.is_collision_free(nearest, target_x, target_y):
                    new_node = Node(target_x, target_y)
                    new_node.parent = nearest
                    new_node.cost = nearest.cost + distance
                    tree.append(new_node)
                    return new_node, True
                return None, False
            new_x, new_y = self.steer(nearest, target_x, target_y)
            if not self.is_collision_free(nearest, new_x, new_y):
                return None, False
            new_node = Node(new_x, new_y)
            near_nodes = self.find_near_nodes_in_tree(tree, new_node)
            best_parent = self.choose_parent_in_tree(tree, new_node, near_nodes)
            if best_parent is None:
                new_node.parent = nearest
                new_node.cost = nearest.cost + self.distance(nearest, new_node)
            tree.append(new_node)
            self.rewire_tree(tree, new_node, near_nodes)
            dist_to_target = np.sqrt((new_x - target_x) ** 2 + (new_y - target_y) ** 2)
            if dist_to_target < self.step_size * 0.5:
                return new_node, True
        return None, False

    def plan(self):
        print("开始RRT*-Connect路径规划...")
        for i in range(self.max_iter):
            rand_x, rand_y = self.random_sample()
            new_node_start = self.extend_tree_with_optimization(self.start_tree, rand_x, rand_y)
            if new_node_start is not None:
                connect_node, success = self.connect_tree_with_optimization(
                    self.goal_tree,
                    new_node_start.x,
                    new_node_start.y,
                )
                if success:
                    print(f"找到路径！迭代次数: {i+1}")
                    self.connect_node_start = new_node_start
                    self.connect_node_goal = connect_node
                    self.nodes = self.start_tree + self.goal_tree
                    return self.extract_dual_path()
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.max_iter}...")
        print("未找到路径，达到最大迭代次数")
        self.nodes = self.start_tree + self.goal_tree
        return None
