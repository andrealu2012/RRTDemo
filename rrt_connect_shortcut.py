"""
RRT-Connect + Path Shortcutting算法模块
"""

import numpy as np
from rrt_connect import RRTConnect


class RRTConnectShortcut(RRTConnect):
    """RRT-Connect + Path Shortcutting算法实现"""

    def __init__(self, start, goal, obstacles, bounds, step_size=0.5, max_iter=1000, random_seed=None):
        super().__init__(start, goal, obstacles, bounds, step_size, max_iter, random_seed)

    def is_direct_connection_free(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        steps = int(distance / 0.1)
        steps = max(steps, 1)
        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            if not (self.bounds[0] <= x <= self.bounds[1] and self.bounds[2] <= y <= self.bounds[3]):
                return False
            for obs in self.obstacles:
                obs_x, obs_y, obs_w, obs_h = obs
                if obs_x <= x <= obs_x + obs_w and obs_y <= y <= obs_y + obs_h:
                    return False
        return True

    def path_shortcutting(self, path, max_iterations=100):
        if path is None or len(path) <= 2:
            return path
        print("开始路径快捷优化...")
        original_length = self.calculate_path_length(path)
        print(f"原始路径: {len(path)}个点, 长度={original_length:.2f}")
        optimized_path = list(path)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            i = 0
            while i < len(optimized_path) - 2:
                max_skip = len(optimized_path) - 1 - i
                for skip in range(max_skip, 1, -1):
                    j = i + skip
                    if self.is_direct_connection_free(optimized_path[i], optimized_path[j]):
                        optimized_path = optimized_path[:i+1] + optimized_path[j:]
                        improved = True
                        print(f"  迭代{iteration}: 跳过{skip-1}个点，从点{i}直连到点{j}")
                        break
                i += 1
            if not improved:
                print("  无法继续优化，停止")
        optimized_length = self.calculate_path_length(optimized_path)
        improvement = ((original_length - optimized_length) / original_length) * 100
        print(f"优化后路径: {len(optimized_path)}个点, 长度={optimized_length:.2f}")
        print(f"路径缩短: {improvement:.1f}%")
        return optimized_path

    def calculate_path_length(self, path):
        if path is None or len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            length += np.sqrt(dx**2 + dy**2)
        return length

    def plan(self):
        print("开始RRT-Connect + Path Shortcutting路径规划...")
        print("第1步: 使用RRT-Connect寻找初始路径...")
        path = super().plan()
        if path is None:
            return None
        print(f"\nRRT-Connect找到路径: {len(path)}个点")
        print(f"树节点总数: {len(self.nodes)}个")
        print("\n第2步: 路径快捷优化...")
        optimized_path = self.path_shortcutting(path)
        return optimized_path
