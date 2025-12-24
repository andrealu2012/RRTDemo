"""
RRT-Connect算法模块
"""

import numpy as np
from rrt_base import Node, RRT


class RRTConnect(RRT):
    """RRT-Connect算法实现 - 双向RRT"""

    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, random_seed=None):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter, random_seed)
        self.start_tree = [self.start]
        self.goal_tree = [Node(goal[0], goal[1])]
        self.connect_node_start = None
        self.connect_node_goal = None

    def nearest_node_in_tree(self, tree, x, y):
        distances = [(node.x - x) ** 2 + (node.y - y) ** 2 for node in tree]
        nearest_idx = np.argmin(distances)
        return tree[nearest_idx]

    def extend_tree(self, tree, rand_x, rand_y):
        nearest = self.nearest_node_in_tree(tree, rand_x, rand_y)
        new_x, new_y = self.steer(nearest, rand_x, rand_y)
        if not self.is_collision_free(nearest, new_x, new_y):
            return None
        new_node = Node(new_x, new_y)
        new_node.parent = nearest
        tree.append(new_node)
        return new_node

    def connect_tree(self, tree, target_x, target_y):
        while True:
            nearest = self.nearest_node_in_tree(tree, target_x, target_y)
            dx = target_x - nearest.x
            dy = target_y - nearest.y
            distance = np.sqrt(dx**2 + dy**2)
            if distance < self.step_size:
                if self.is_collision_free(nearest, target_x, target_y):
                    new_node = Node(target_x, target_y)
                    new_node.parent = nearest
                    tree.append(new_node)
                    return new_node, True
                return None, False
            new_x, new_y = self.steer(nearest, target_x, target_y)
            if not self.is_collision_free(nearest, new_x, new_y):
                return None, False
            new_node = Node(new_x, new_y)
            new_node.parent = nearest
            tree.append(new_node)
            dist_to_target = np.sqrt((new_x - target_x) ** 2 + (new_y - target_y) ** 2)
            if dist_to_target < self.step_size * 0.5:
                return new_node, True

    def extract_dual_path(self):
        path_start = []
        current = self.connect_node_start
        while current is not None:
            path_start.append((current.x, current.y))
            current = current.parent
        path_start = path_start[::-1]
        path_goal = []
        current = self.connect_node_goal.parent
        while current is not None:
            path_goal.append((current.x, current.y))
            current = current.parent
        return path_start + path_goal

    def plan(self):
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            print(f"设置随机种子: {self.random_seed}")
        print("开始RRT-Connect路径规划...")
        for i in range(self.max_iter):
            rand_x, rand_y = self.random_sample()
            new_node_start = self.extend_tree(self.start_tree, rand_x, rand_y)
            if new_node_start is not None:
                connect_node, success = self.connect_tree(
                    self.goal_tree,
                    new_node_start.x,
                    new_node_start.y,
                )
                if success:
                    print(f"找到路径！迭代次数: {i+1}")
                    self.connect_node_start = new_node_start
                    self.connect_node_goal = connect_node
                    self.nodes = self.start_tree + self.goal_tree
                    print(f"树节点总数: {len(self.nodes)}个 (起始树: {len(self.start_tree)}, 目标树: {len(self.goal_tree)})")
                    return self.extract_dual_path()
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.max_iter}...")
        print("未找到路径，达到最大迭代次数")
        self.nodes = self.start_tree + self.goal_tree
        return None
