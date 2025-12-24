"""
Base RRT 实现和通用节点定义
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Node:
    """RRT树的节点"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0  # 从起点到该节点的代价（用于RRT*）


class RRT:
    """RRT算法实现"""

    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, random_seed=None):
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.obstacles = obstacles
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_threshold = 0.5
        self.nodes = [self.start]
        self.random_seed = random_seed

    def random_sample(self):
        if np.random.random() < 0.1:
            return self.goal.x, self.goal.y
        x = np.random.uniform(self.bounds[0], self.bounds[1])
        y = np.random.uniform(self.bounds[2], self.bounds[3])
        return x, y

    def nearest_node(self, x, y):
        distances = [(node.x - x) ** 2 + (node.y - y) ** 2 for node in self.nodes]
        nearest_idx = np.argmin(distances)
        return self.nodes[nearest_idx]

    def steer(self, from_node, to_x, to_y):
        dx = to_x - from_node.x
        dy = to_y - from_node.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < self.step_size:
            return to_x, to_y

        ratio = self.step_size / distance
        new_x = from_node.x + dx * ratio
        new_y = from_node.y + dy * ratio
        return new_x, new_y

    def is_collision_free(self, from_node, to_x, to_y):
        steps = int(np.sqrt((to_x - from_node.x) ** 2 + (to_y - from_node.y) ** 2) / 0.1)
        steps = max(steps, 1)
        for i in range(steps + 1):
            t = i / steps
            x = from_node.x + t * (to_x - from_node.x)
            y = from_node.y + t * (to_y - from_node.y)
            if not (self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3]):
                return False
            for obs in self.obstacles:
                obs_x, obs_y, obs_w, obs_h = obs
                if obs_x <= x <= obs_x + obs_w and obs_y <= y <= obs_y + obs_h:
                    return False
        return True

    def is_goal_reached(self, node):
        distance = np.sqrt((node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2)
        return distance < self.goal_threshold

    def plan(self):
        print("开始RRT路径规划...")
        for i in range(self.max_iter):
            rand_x, rand_y = self.random_sample()
            nearest = self.nearest_node(rand_x, rand_y)
            new_x, new_y = self.steer(nearest, rand_x, rand_y)
            if not self.is_collision_free(nearest, new_x, new_y):
                continue
            new_node = Node(new_x, new_y)
            new_node.parent = nearest
            self.nodes.append(new_node)
            if self.is_goal_reached(new_node):
                print(f"找到路径！迭代次数: {i+1}")
                self.goal.parent = new_node
                return self.extract_path()
            if (i + 1) % 100 == 0:
                print(f"迭代 {i+1}/{self.max_iter}...")
        print("未找到路径，达到最大迭代次数")
        return None

    def extract_path(self):
        path = []
        current = self.goal
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        return path[::-1]

    def visualize(self, path=None):
        fig, ax = plt.subplots(figsize=(10, 10))
        for obs in self.obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            rect = patches.Rectangle((obs_x, obs_y), obs_w, obs_h,
                                     linewidth=1, edgecolor='black',
                                     facecolor='gray', alpha=0.7)
            ax.add_patch(rect)
        for node in self.nodes:
            if node.parent is not None:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'c-', linewidth=0.5, alpha=0.3)
        node_x = [node.x for node in self.nodes]
        node_y = [node.y for node in self.nodes]
        ax.plot(node_x, node_y, 'c.', markersize=2, alpha=0.5)
        ax.plot(self.start.x, self.start.y, 'go', markersize=15, label='起点')
        ax.plot(self.goal.x, self.goal.y, 'r*', markersize=20, label='终点')
        if path is not None:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, label='规划路径')
            path_length = sum(np.sqrt((path[i][0] - path[i-1][0])**2 +
                                     (path[i][1] - path[i-1][1])**2)
                              for i in range(1, len(path)))
            ax.text(0.02, 0.98, f'路径长度: {path_length:.2f}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('RRT 路径规划演示', fontsize=16)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.tight_layout()
        plt.show()
