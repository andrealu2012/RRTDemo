"""
交互式RRT算法演示窗口
"""

import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from rrt_base import RRT
from rrt_connect import RRTConnect
from rrt_connect_shortcut import RRTConnectShortcut
from rrt_star import RRTStar
from rrt_star_connect import RRTStarConnect


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False


class InteractiveVisualizer:
    """交互式可视化工具，可以切换显示所有RRT算法变体"""

    def __init__(self, start, goal, bounds, step_size, max_iter, search_radius,
                 rrt, rrt_star, rrt_connect, rrt_star_connect, rrt_connect_shortcut,
                 rrt_path, rrt_star_path, rrt_connect_path, rrt_star_connect_path, rrt_connect_shortcut_path,
                 rrt_time, rrt_star_time, rrt_connect_time, rrt_star_connect_time, rrt_connect_shortcut_time):
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.step_size = step_size
        self.max_iter = max_iter
        self.search_radius = search_radius

        self.rrt = rrt
        self.rrt_star = rrt_star
        self.rrt_connect = rrt_connect
        self.rrt_star_connect = rrt_star_connect
        self.rrt_connect_shortcut = rrt_connect_shortcut
        self.rrt_path = rrt_path
        self.rrt_star_path = rrt_star_path
        self.rrt_connect_path = rrt_connect_path
        self.rrt_star_connect_path = rrt_star_connect_path
        self.rrt_connect_shortcut_path = rrt_connect_shortcut_path

        self.rrt_time = rrt_time
        self.rrt_star_time = rrt_star_time
        self.rrt_connect_time = rrt_connect_time
        self.rrt_star_connect_time = rrt_star_connect_time
        self.rrt_connect_shortcut_time = rrt_connect_shortcut_time

        # 障碍物数量范围
        self.min_obstacles = 4
        self.max_obstacles = 8

        self.current_algorithm = 'rrt'
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        plt.subplots_adjust(bottom=0.12, right=0.83)

        # 创建五个算法切换按钮（右侧垂直布局，紧凑）
        ax_rrt = plt.axes([0.84, 0.50, 0.13, 0.045])
        ax_rrt_star = plt.axes([0.84, 0.44, 0.13, 0.045])
        ax_rrt_connect = plt.axes([0.84, 0.38, 0.13, 0.045])
        ax_rrt_star_connect = plt.axes([0.84, 0.28, 0.13, 0.045])
        ax_rrt_connect_shortcut = plt.axes([0.84, 0.22, 0.13, 0.045])

        self.btn_rrt = Button(ax_rrt, 'RRT', color='lightgray', hovercolor='darkgray')
        self.btn_rrt_star = Button(ax_rrt_star, 'RRT*', color='lightgray', hovercolor='darkgray')
        self.btn_rrt_connect = Button(ax_rrt_connect, 'RRT-Connect', color='lightgray', hovercolor='darkgray')
        self.btn_rrt_star_connect = Button(ax_rrt_star_connect, 'RRT*-Connect', color='lightgray', hovercolor='darkgray')
        self.btn_rrt_connect_shortcut = Button(ax_rrt_connect_shortcut, 'RRT-C+Shortcut', color='lightgray', hovercolor='darkgray')

        self.btn_rrt.on_clicked(self.show_rrt)
        self.btn_rrt_star.on_clicked(self.show_rrt_star)
        self.btn_rrt_connect.on_clicked(self.show_rrt_connect)
        self.btn_rrt_star_connect.on_clicked(self.show_rrt_star_connect)
        self.btn_rrt_connect_shortcut.on_clicked(self.show_rrt_connect_shortcut)

        # 创建生成障碍物按钮（右侧算法按钮下方）
        ax_generate = plt.axes([0.84, 0.14, 0.13, 0.06])
        self.btn_generate = Button(ax_generate, '生成新\n障碍物', 
                                   color='lightgray', hovercolor='darkgray')
        self.btn_generate.label.set_fontsize(10)
        self.btn_generate.label.set_fontweight('bold')
        self.btn_generate.on_clicked(self.generate_new_obstacles)

        # 创建障碍物数量范围的滑杆控件（底部居中）
        ax_min_obs = plt.axes([0.25, 0.065, 0.5, 0.02])
        ax_max_obs = plt.axes([0.25, 0.03, 0.5, 0.02])
        
        self.slider_min = Slider(ax_min_obs, '最小障碍物', 1, 15, valinit=self.min_obstacles, 
                                valstep=1, color='black')
        self.slider_max = Slider(ax_max_obs, '最大障碍物', 1, 15, valinit=self.max_obstacles, 
                                valstep=1, color='black')
        
        self.slider_min.on_changed(self.update_min_obstacles)
        self.slider_max.on_changed(self.update_max_obstacles)

        self.draw_algorithm()

    def clear_plot(self):
        self.ax.clear()

    def draw_obstacles(self):
        for obs in self.rrt.obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            rect = patches.Rectangle((obs_x, obs_y), obs_w, obs_h,
                                     linewidth=1, edgecolor='black',
                                     facecolor='gray', alpha=0.7)
            self.ax.add_patch(rect)

    def draw_algorithm(self):
        self.clear_plot()
        if self.current_algorithm == 'rrt':
            algorithm = self.rrt
            path = self.rrt_path
            title = 'RRT 路径规划'
            color = 'c'
            exec_time = self.rrt_time
        elif self.current_algorithm == 'rrt_star':
            algorithm = self.rrt_star
            path = self.rrt_star_path
            title = 'RRT* 路径规划'
            color = 'm'
            exec_time = self.rrt_star_time
        elif self.current_algorithm == 'rrt_connect':
            algorithm = self.rrt_connect
            path = self.rrt_connect_path
            title = 'RRT-Connect 路径规划'
            color = 'orange'
            exec_time = self.rrt_connect_time
        elif self.current_algorithm == 'rrt_star_connect':
            algorithm = self.rrt_star_connect
            path = self.rrt_star_connect_path
            title = 'RRT*-Connect 路径规划'
            color = 'purple'
            exec_time = self.rrt_star_connect_time
        else:
            algorithm = self.rrt_connect_shortcut
            path = self.rrt_connect_shortcut_path
            title = 'RRT-Connect + Path Shortcutting'
            color = 'brown'
            exec_time = self.rrt_connect_shortcut_time

        self.draw_obstacles()
        for node in algorithm.nodes:
            if node.parent is not None:
                self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y],
                             color=color, linewidth=0.5, alpha=0.3)
        node_x = [node.x for node in algorithm.nodes]
        node_y = [node.y for node in algorithm.nodes]
        self.ax.plot(node_x, node_y, color=color, marker='.',
                     markersize=2, linestyle='', alpha=0.5)
        self.ax.plot(algorithm.start.x, algorithm.start.y, 'go', markersize=15, label='起点')
        self.ax.plot(algorithm.goal.x, algorithm.goal.y, 'r*', markersize=20, label='终点')

        if path is not None:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            self.ax.plot(path_x, path_y, 'b-', linewidth=3, label='规划路径')
            path_length = sum(
                np.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
                for i in range(1, len(path))
            )
            info_text = (
                f'算法: {title}\n'
                f'总节点数: {len(algorithm.nodes)}\n'
                f'路径节点数: {len(path)}\n'
                f'路径长度: {path_length:.2f}\n'
                f'计算时间: {exec_time:.3f}秒'
            )
        else:
            info_text = (
                f'算法: {title}\n'
                f'总节点数: {len(algorithm.nodes)}\n'
                f'未找到路径\n'
                f'计算时间: {exec_time:.3f}秒'
            )
        self.ax.text(0.02, 0.98, info_text,
                     transform=self.ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                     fontsize=10)
        self.ax.set_xlim(algorithm.bounds[0], algorithm.bounds[1])
        self.ax.set_ylim(algorithm.bounds[2], algorithm.bounds[3])
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='upper right')
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.fig.canvas.draw()

    def show_rrt(self, event):
        self.current_algorithm = 'rrt'
        self.draw_algorithm()

    def show_rrt_star(self, event):
        self.current_algorithm = 'rrt_star'
        self.draw_algorithm()

    def show_rrt_connect(self, event):
        self.current_algorithm = 'rrt_connect'
        self.draw_algorithm()

    def show_rrt_star_connect(self, event):
        self.current_algorithm = 'rrt_star_connect'
        self.draw_algorithm()

    def show_rrt_connect_shortcut(self, event):
        self.current_algorithm = 'rrt_connect_shortcut'
        self.draw_algorithm()

    def update_min_obstacles(self, val):
        """更新最小障碍物数量"""
        self.min_obstacles = int(val)
        # 确保最小值不大于最大值
        if self.min_obstacles > self.max_obstacles:
            self.max_obstacles = self.min_obstacles
            self.slider_max.set_val(self.max_obstacles)
    
    def update_max_obstacles(self, val):
        """更新最大障碍物数量"""
        self.max_obstacles = int(val)
        # 确保最大值不小于最小值
        if self.max_obstacles < self.min_obstacles:
            self.min_obstacles = self.max_obstacles
            self.slider_min.set_val(self.min_obstacles)

    def generate_random_obstacles(self, num_obstacles=5):
        obstacles = []
        x_range = self.bounds[1] - self.bounds[0]
        y_range = self.bounds[3] - self.bounds[2]
        start_x, start_y = self.start
        goal_x, goal_y = self.goal
        attempts = 0
        max_attempts = 100
        while len(obstacles) < num_obstacles and attempts < max_attempts:
            attempts += 1
            obs_w = np.random.uniform(0.5, 2.0)
            obs_h = np.random.uniform(0.5, 2.0)
            obs_x = np.random.uniform(self.bounds[0] + 0.5, self.bounds[1] - obs_w - 0.5)
            obs_y = np.random.uniform(self.bounds[2] + 0.5, self.bounds[3] - obs_h - 0.5)
            start_dist = np.sqrt((obs_x + obs_w / 2 - start_x) ** 2 + (obs_y + obs_h / 2 - start_y) ** 2)
            goal_dist = np.sqrt((obs_x + obs_w / 2 - goal_x) ** 2 + (obs_y + obs_h / 2 - goal_y) ** 2)
            if start_dist > 1.5 and goal_dist > 1.5:
                overlap = False
                for existing_obs in obstacles:
                    ex_x, ex_y, ex_w, ex_h = existing_obs
                    if not (obs_x + obs_w < ex_x or obs_x > ex_x + ex_w or
                            obs_y + obs_h < ex_y or obs_y > ex_y + ex_h):
                        overlap = True
                        break
                if not overlap:
                    obstacles.append((obs_x, obs_y, obs_w, obs_h))
        return obstacles

    def generate_new_obstacles(self, event):
        print("\n" + "=" * 70)
        print("生成新障碍物并重新规划...")
        print("=" * 70)
        obstacles = self.generate_random_obstacles(num_obstacles=np.random.randint(self.min_obstacles, self.max_obstacles + 1))
        print(f"\n生成了 {len(obstacles)} 个障碍物 (范围: {self.min_obstacles}-{self.max_obstacles})")
        print("\n[1/5] 运行RRT算法...")
        start_time = time.time()
        self.rrt = RRT(self.start, self.goal, obstacles, self.bounds,
                       self.step_size, self.max_iter)
        self.rrt_path = self.rrt.plan()
        self.rrt_time = time.time() - start_time
        if self.rrt_path:
            rrt_length = sum(
                np.sqrt((self.rrt_path[i][0] - self.rrt_path[i-1][0])**2 + (self.rrt_path[i][1] - self.rrt_path[i-1][1])**2)
                for i in range(1, len(self.rrt_path))
            )
            print(f"RRT结果: 节点数={len(self.rrt.nodes)}, 路径长度={rrt_length:.2f}, 时间={self.rrt_time:.3f}秒")
        else:
            print(f"RRT结果: 未找到路径, 时间={self.rrt_time:.3f}秒")
        print("\n[2/5] 运行RRT*算法...")
        start_time = time.time()
        self.rrt_star = RRTStar(self.start, self.goal, obstacles, self.bounds,
                                self.step_size, self.max_iter, self.search_radius)
        self.rrt_star_path = self.rrt_star.plan()
        self.rrt_star_time = time.time() - start_time
        if self.rrt_star_path:
            rrt_star_length = sum(
                np.sqrt((self.rrt_star_path[i][0] - self.rrt_star_path[i-1][0])**2 + (self.rrt_star_path[i][1] - self.rrt_star_path[i-1][1])**2)
                for i in range(1, len(self.rrt_star_path))
            )
            print(f"RRT*结果: 节点数={len(self.rrt_star.nodes)}, 路径长度={rrt_star_length:.2f}, 时间={self.rrt_star_time:.3f}秒")
        else:
            print(f"RRT*结果: 未找到路径, 时间={self.rrt_star_time:.3f}秒")
        print("\n[3/5] 运行RRT-Connect算法...")
        start_time = time.time()
        self.rrt_connect = RRTConnect(self.start, self.goal, obstacles, self.bounds,
                                      self.step_size, self.max_iter)
        self.rrt_connect_path = self.rrt_connect.plan()
        self.rrt_connect_time = time.time() - start_time
        if self.rrt_connect_path:
            rrt_connect_length = sum(
                np.sqrt((self.rrt_connect_path[i][0] - self.rrt_connect_path[i-1][0])**2 + (self.rrt_connect_path[i][1] - self.rrt_connect_path[i-1][1])**2)
                for i in range(1, len(self.rrt_connect_path))
            )
            print(f"RRT-Connect结果: 节点数={len(self.rrt_connect.nodes)}, 路径长度={rrt_connect_length:.2f}, 时间={self.rrt_connect_time:.3f}秒")
        else:
            print(f"RRT-Connect结果: 未找到路径, 时间={self.rrt_connect_time:.3f}秒")
        print("\n[4/5] 运行RRT*-Connect算法...")
        start_time = time.time()
        self.rrt_star_connect = RRTStarConnect(self.start, self.goal, obstacles, self.bounds,
                                               self.step_size, self.max_iter, self.search_radius)
        self.rrt_star_connect_path = self.rrt_star_connect.plan()
        self.rrt_star_connect_time = time.time() - start_time
        if self.rrt_star_connect_path:
            rrt_star_connect_length = sum(
                np.sqrt((self.rrt_star_connect_path[i][0] - self.rrt_star_connect_path[i-1][0])**2 + (self.rrt_star_connect_path[i][1] - self.rrt_star_connect_path[i-1][1])**2)
                for i in range(1, len(self.rrt_star_connect_path))
            )
            print(f"RRT*-Connect结果: 节点数={len(self.rrt_star_connect.nodes)}, 路径长度={rrt_star_connect_length:.2f}, 时间={self.rrt_star_connect_time:.3f}秒")
        else:
            print(f"RRT*-Connect结果: 未找到路径, 时间={self.rrt_star_connect_time:.3f}秒")
        print("\n[5/5] 运行RRT-Connect + Path Shortcutting算法...")
        start_time = time.time()
        self.rrt_connect_shortcut = RRTConnectShortcut(self.start, self.goal, obstacles, self.bounds,
                                                       self.step_size, self.max_iter)
        self.rrt_connect_shortcut_path = self.rrt_connect_shortcut.plan()
        self.rrt_connect_shortcut_time = time.time() - start_time
        if self.rrt_connect_shortcut_path:
            shortcut_length = sum(
                np.sqrt((self.rrt_connect_shortcut_path[i][0] - self.rrt_connect_shortcut_path[i-1][0])**2 +
                        (self.rrt_connect_shortcut_path[i][1] - self.rrt_connect_shortcut_path[i-1][1])**2)
                for i in range(1, len(self.rrt_connect_shortcut_path))
            )
            print(f"RRT-Connect+Shortcut结果: 节点数={len(self.rrt_connect_shortcut.nodes)}, 路径长度={shortcut_length:.2f}, 时间={self.rrt_connect_shortcut_time:.3f}秒")
        else:
            print(f"RRT-Connect+Shortcut结果: 未找到路径, 时间={self.rrt_connect_shortcut_time:.3f}秒")
        print("\n重新规划完成！")
        print("=" * 70)
        self.draw_algorithm()

    def show(self):
        plt.show()


def main():
    start = (1, 1)
    goal = (9, 9)
    obstacles = [
        (3, 2, 1, 3),
        (5, 5, 2, 2),
        (2, 6, 2, 1),
        (7, 3, 1, 4),
    ]
    bounds = (0, 10, 0, 10)

    print("=" * 70)
    print("RRT算法全家桶对比演示")
    print("=" * 70)

    print("\n[1/5] 运行RRT算法...")
    start_time = time.time()
    rrt = RRT(start, goal, obstacles, bounds, step_size=0.5, max_iter=2000)
    rrt_path = rrt.plan()
    rrt_time = time.time() - start_time
    rrt_length = None
    if rrt_path:
        rrt_length = sum(
            np.sqrt((rrt_path[i][0] - rrt_path[i-1][0])**2 + (rrt_path[i][1] - rrt_path[i-1][1])**2)
            for i in range(1, len(rrt_path))
        )
        print(f"RRT结果: 节点数={len(rrt.nodes)}, 路径长度={rrt_length:.2f}, 时间={rrt_time:.3f}秒")
    else:
        print(f"RRT结果: 未找到路径, 时间={rrt_time:.3f}秒")

    print("\n[2/5] 运行RRT*算法...")
    start_time = time.time()
    rrt_star = RRTStar(start, goal, obstacles, bounds, step_size=0.5, max_iter=2000, search_radius=1.5)
    rrt_star_path = rrt_star.plan()
    rrt_star_time = time.time() - start_time
    rrt_star_length = None
    if rrt_star_path:
        rrt_star_length = sum(
            np.sqrt((rrt_star_path[i][0] - rrt_star_path[i-1][0])**2 + (rrt_star_path[i][1] - rrt_star_path[i-1][1])**2)
            for i in range(1, len(rrt_star_path))
        )
        print(f"RRT*结果: 节点数={len(rrt_star.nodes)}, 路径长度={rrt_star_length:.2f}, 时间={rrt_star_time:.3f}秒")
    else:
        print(f"RRT*结果: 未找到路径, 时间={rrt_star_time:.3f}秒")

    print("\n[3/5] 运行RRT-Connect算法...")
    start_time = time.time()
    rrt_connect = RRTConnect(start, goal, obstacles, bounds, step_size=0.5, max_iter=2000)
    rrt_connect_path = rrt_connect.plan()
    rrt_connect_time = time.time() - start_time
    rrt_connect_length = None
    if rrt_connect_path:
        rrt_connect_length = sum(
            np.sqrt((rrt_connect_path[i][0] - rrt_connect_path[i-1][0])**2 + (rrt_connect_path[i][1] - rrt_connect_path[i-1][1])**2)
            for i in range(1, len(rrt_connect_path))
        )
        print(f"RRT-Connect结果: 节点数={len(rrt_connect.nodes)}, 路径长度={rrt_connect_length:.2f}, 时间={rrt_connect_time:.3f}秒")
    else:
        print(f"RRT-Connect结果: 未找到路径, 时间={rrt_connect_time:.3f}秒")

    print("\n[4/5] 运行RRT*-Connect算法...")
    start_time = time.time()
    rrt_star_connect = RRTStarConnect(start, goal, obstacles, bounds, step_size=0.5, max_iter=2000, search_radius=1.5)
    rrt_star_connect_path = rrt_star_connect.plan()
    rrt_star_connect_time = time.time() - start_time
    rrt_star_connect_length = None
    if rrt_star_connect_path:
        rrt_star_connect_length = sum(
            np.sqrt((rrt_star_connect_path[i][0] - rrt_star_connect_path[i-1][0])**2 + (rrt_star_connect_path[i][1] - rrt_star_connect_path[i-1][1])**2)
            for i in range(1, len(rrt_star_connect_path))
        )
        print(f"RRT*-Connect结果: 节点数={len(rrt_star_connect.nodes)}, 路径长度={rrt_star_connect_length:.2f}, 时间={rrt_star_connect_time:.3f}秒")
    else:
        print(f"RRT*-Connect结果: 未找到路径, 时间={rrt_star_connect_time:.3f}秒")

    print("\n[5/5] 运行RRT-Connect + Path Shortcutting算法...")
    start_time = time.time()
    rrt_connect_shortcut = RRTConnectShortcut(start, goal, obstacles, bounds, step_size=0.5, max_iter=2000)
    rrt_connect_shortcut_path = rrt_connect_shortcut.plan()
    rrt_connect_shortcut_time = time.time() - start_time
    rrt_connect_shortcut_length = None
    if rrt_connect_shortcut_path:
        rrt_connect_shortcut_length = sum(
            np.sqrt((rrt_connect_shortcut_path[i][0] - rrt_connect_shortcut_path[i-1][0])**2 + (rrt_connect_shortcut_path[i][1] - rrt_connect_shortcut_path[i-1][1])**2)
            for i in range(1, len(rrt_connect_shortcut_path))
        )
        print(f"RRT-Connect+Shortcut结果: 节点数={len(rrt_connect_shortcut.nodes)}, 路径长度={rrt_connect_shortcut_length:.2f}, 时间={rrt_connect_shortcut_time:.3f}秒")
    else:
        print(f"RRT-Connect+Shortcut结果: 未找到路径, 时间={rrt_connect_shortcut_time:.3f}秒")

    print("\n" + "=" * 70)
    print("算法对比:")
    print("=" * 70)
    if rrt_length:
        print(f"RRT:                     路径长度={rrt_length:.2f}, 节点数={len(rrt.nodes)}, 时间={rrt_time:.3f}秒")
    else:
        print(f"RRT:                     未找到路径, 节点数={len(rrt.nodes)}, 时间={rrt_time:.3f}秒")
    if rrt_star_length:
        print(f"RRT*:                    路径长度={rrt_star_length:.2f}, 节点数={len(rrt_star.nodes)}, 时间={rrt_star_time:.3f}秒")
    else:
        print(f"RRT*:                    未找到路径, 节点数={len(rrt_star.nodes)}, 时间={rrt_star_time:.3f}秒")
    if rrt_connect_length:
        print(f"RRT-Connect:             路径长度={rrt_connect_length:.2f}, 节点数={len(rrt_connect.nodes)}, 时间={rrt_connect_time:.3f}秒")
    else:
        print(f"RRT-Connect:             未找到路径, 节点数={len(rrt_connect.nodes)}, 时间={rrt_connect_time:.3f}秒")
    if rrt_star_connect_length:
        print(f"RRT*-Connect:            路径长度={rrt_star_connect_length:.2f}, 节点数={len(rrt_star_connect.nodes)}, 时间={rrt_star_connect_time:.3f}秒")
    else:
        print(f"RRT*-Connect:            未找到路径, 节点数={len(rrt_star_connect.nodes)}, 时间={rrt_star_connect_time:.3f}秒")
    if rrt_connect_shortcut_length:
        print(f"RRT-Connect+Shortcut:    路径长度={rrt_connect_shortcut_length:.2f}, 节点数={len(rrt_connect_shortcut.nodes)}, 时间={rrt_connect_shortcut_time:.3f}秒")
    else:
        print(f"RRT-Connect+Shortcut:    未找到路径, 节点数={len(rrt_connect_shortcut.nodes)}, 时间={rrt_connect_shortcut_time:.3f}秒")

    print("\n算法特点:")
    print("- RRT:                   快速探索，找到第一条可行路径")
    print("- RRT*:                  路径优化，生成更短更平滑的路径")
    print("- RRT-Connect:           双向搜索，通常最快找到路径")
    print("- RRT*-Connect:          结合双向搜索和路径优化")
    print("- RRT-Connect+Shortcut:  快速找路径后进行路径快捷优化")

    print("\n提示: 在可视化窗口中点击按钮可以切换显示不同算法的结果")
    print("=" * 70)

    visualizer = InteractiveVisualizer(
        start, goal, bounds, 0.5, 2000, 1.5,
        rrt, rrt_star, rrt_connect, rrt_star_connect, rrt_connect_shortcut,
        rrt_path, rrt_star_path, rrt_connect_path, rrt_star_connect_path, rrt_connect_shortcut_path,
        rrt_time, rrt_star_time, rrt_connect_time, rrt_star_connect_time, rrt_connect_shortcut_time,
    )
    visualizer.show()


if __name__ == "__main__":
    main()
