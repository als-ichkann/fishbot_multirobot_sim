"""
ROVER_3D ：主入口与规划进程
PlanningAPFProcess: GMM 宏观规划 + APF 轨迹生成，单步式接口供 ROS2 节点调用。
与 MPC 分离：仅发布轨迹，控制由 mpc_control 订阅执行。
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import numpy as np
from scipy.stats import multivariate_normal

# 依赖：init_Graph_CVaR_3D, init_scene_3D
from rover3d_navigation import init_Graph_CVaR_3D
from rover3d_navigation import init_scene_3D

try:
    from . import control_law_3D
    from . import Planning_3D
except ImportError:
    import control_law_3D
    import Planning_3D
import networkx as nx


def _mean_to_key(m) -> tuple:
    """将均值转为可哈希的 tuple，用于索引匹配。"""
    arr = np.asarray(m).flatten()
    return tuple(float(x) for x in arr[:3])


def _find_mean_index(means_list, mean) -> int:
    """在 means_list 中查找与 mean 匹配的索引（坐标容差匹配）。"""
    key = _mean_to_key(mean)
    for i, m in enumerate(means_list):
        if np.allclose(np.asarray(m).flatten()[:3], np.asarray(key)):
            return i
    raise ValueError(f"Mean {mean} not found in means_list")


def _adj_to_graph(adj: np.ndarray, n: int) -> "nx.DiGraph":
    """将邻接矩阵转为 networkx 有向图。"""
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i, j] != 0 and not np.isnan(adj[i, j]):
                G.add_edge(i, j, weight=float(adj[i, j]))
    return G


class PlanningAPFProcess:
    """
    ROS2 兼容规划进程：单步式 run_one_cycle 接口。
    无多进程、无 MPC：仅做 GMM 优化 + APF 轨迹生成。
    """

    def __init__(
        self,
        num_robots: int,
        esdf_map,
        xa: float,
        xb: float,
        ya: float,
        yb: float,
        za: float,
        zb: float,
        goal_means: list,
        goal_covs: list,
        goal_weights: list,
        gmm_interp_steps: int = 5,
        max_apf_try: int = 10,
        use_gmm_trajectory_slp: bool = True,
        grid_step: float = 2.0,
        config_dir: Optional[str] = None,
    ):
        self.num_robots = num_robots
        self.esdf_map = esdf_map
        self.xa, self.xb = xa, xb
        self.ya, self.yb = ya, yb
        self.za, self.zb = za, zb
        self.fmeans = [list(m) for m in goal_means]
        self.fcovs = [np.asarray(c).reshape(3, 3) if np.size(c) == 9 else c for c in goal_covs]
        self.fweights = list(goal_weights)
        self.gmm_interp_steps = gmm_interp_steps
        self.max_apf_try = max_apf_try
        self.use_gmm_trajectory_slp = use_gmm_trajectory_slp
        self.alpha = 0.05

        # 优先从 config 加载预计算的高斯节点与权重表
        loaded_from_config = False
        if config_dir:
            path_means = os.path.join(config_dir, "GC_means_3D.json")
            path_covs = os.path.join(config_dir, "GC_covs_3D.json")
            if os.path.exists(path_means) and os.path.exists(path_covs):
                with open(path_means, "r") as f:
                    self.GC_means = json.load(f)
                with open(path_covs, "r") as f:
                    raw_covs = json.load(f)
                self.GC_covs = [
                    np.asarray(c).reshape(3, 3) if np.size(c) == 9 else np.asarray(c)
                    for c in raw_covs
                ]
                loaded_from_config = True

        if not loaded_from_config:
            # 基于地图边界构建高斯节点，grid_step 越大节点越少、初始化越快
            step = float(grid_step)
            mean_table = []
            for i in np.arange(xa, xb, step):
                for j in np.arange(ya, yb, step):
                    for k in np.arange(za, zb, step):
                        mean_table.append([float(i + 0.5), float(j + 0.5), float(k + 0.5)])
            if len(mean_table) == 0:
                mean_table = [[(xa + xb) / 2, (ya + yb) / 2, (za + zb) / 2]]
            self.GC_means, self.GC_covs = init_Graph_CVaR_3D.init_GC_Nodes(mean_table)

        self.conbinedmeans_list = self.fmeans + self.GC_means
        self.conbinedcovs_list = self.fcovs + self.GC_covs
        Numnode = len(self.conbinedmeans_list)

        # Wasserstein 与 Node_PDF 表：优先从 config 加载
        path_w = os.path.join(config_dir or "", "Wasserstein_table_3D.npy")
        path_pdf = os.path.join(config_dir or "", "Node_PDF_table_3D.npy")
        if not os.path.exists(path_w):
            path_w = os.path.join(config_dir or "", "Wasserstein_table_3D.npy")
        if not os.path.exists(path_pdf):
            path_pdf = os.path.join(config_dir or "", "Node_PDF_table_3D.npy")
        if config_dir and os.path.exists(path_w) and os.path.exists(path_pdf):
            w_load = np.load(path_w)
            pdf_load = np.load(path_pdf)
            if w_load.shape == (Numnode, Numnode) and pdf_load.shape == (Numnode, Numnode):
                self.Wasserstein_table = w_load
                self.Node_PDF_table = pdf_load
            else:
                # 维度不匹配，重新计算
                self._compute_wasserstein_pdf_tables(Numnode)
        else:
            self._compute_wasserstein_pdf_tables(Numnode)

        # Graph_GC (最短路径长度矩阵)：优先从 config 加载
        path_graph = os.path.join(config_dir or "", "Graph_GC_3D.npy")
        if config_dir and os.path.exists(path_graph):
            graph_load = np.load(path_graph)
            if graph_load.shape == (Numnode, Numnode):
                _, self.Graph_GC = Planning_3D.shortest_path(
                    _adj_to_graph(graph_load, Numnode)
                )
            else:
                self._build_graph_gc(Numnode)
        else:
            self._build_graph_gc(Numnode)

        # 当前分布：从初始机器人位置估计
        self.current_means = list(self.fmeans)
        self.current_covs = list(self.fcovs)
        self.current_weights = list(self.fweights)
        self.optimization_k = 0
        self.goalFlag = 1
        self.StopFlag = 0
        self.flag = 0
        self.GMM: List[list] = []
        self.WStack: List = []
        self.step = 0
        self.robots_positions_expected: Optional[np.ndarray] = None
        self.stop_flag = False

    def _compute_wasserstein_pdf_tables(self, Numnode: int) -> None:
        """计算 Wasserstein 与 Node_PDF 表。"""
        self.Wasserstein_table = np.zeros((Numnode, Numnode))
        self.Node_PDF_table = np.zeros((Numnode, Numnode))
        for i in range(Numnode):
            for j in range(Numnode):
                m1 = self.conbinedmeans_list[i]
                c1 = self.conbinedcovs_list[i] if hasattr(self.conbinedcovs_list[i], "shape") else np.array(self.conbinedcovs_list[i]).reshape(3, 3)
                m2 = self.conbinedmeans_list[j]
                c2 = self.conbinedcovs_list[j] if hasattr(self.conbinedcovs_list[j], "shape") else np.array(self.conbinedcovs_list[j]).reshape(3, 3)
                self.Wasserstein_table[i, j] = control_law_3D.Wasserstein_distance(m1, c1, m2, c2)
                self.Node_PDF_table[i, j] = multivariate_normal.pdf(m1, mean=np.array(m2), cov=c2)

    def _build_graph_gc(self, Numnode: int) -> None:
        """构建图并计算最短路径。"""
        Graph_adj = init_Graph_CVaR_3D.init_Graph_GC(
            self.conbinedmeans_list, self.conbinedcovs_list, self.Wasserstein_table,
            xa=self.xa, ya=self.ya, za=self.za, xb=self.xb, yb=self.yb, zb=self.zb,
        )
        G = nx.DiGraph()
        for i in range(Numnode):
            G.add_node(i)
        rows, cols = Graph_adj.shape
        for i in range(rows):
            for j in range(cols):
                if Graph_adj[i, j] != 0:
                    G.add_edge(i, j, weight=Graph_adj[i, j])
        _, self.Graph_GC = Planning_3D.shortest_path(G)

    def _gmm_score_samples(self, means, covs, weights, points: np.ndarray) -> np.ndarray:
        """用 GMM 对点集计算 log 概率。"""
        from sklearn.mixture import GaussianMixture
        n_comp = len(means)
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full')
        gmm.means_ = np.array(means)
        gmm.covariances_ = np.array([np.asarray(c).reshape(3, 3) for c in covs])
        gmm.weights_ = np.array(weights)
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(gmm.covariances_))
        return gmm.score_samples(points)

    def run_one_cycle(
        self, robots_positions: np.ndarray
    ) -> Optional[Tuple[List[np.ndarray], bool]]:
        """
        执行一次规划循环。
        :param robots_positions: (N, 3) 当前机器人位置
        :return: (trajectories, stop_flag) 或 None（本步无新轨迹）
        """
        robots_positions = np.asarray(robots_positions, dtype=float).reshape(-1, 3)
        if robots_positions.shape[0] != self.num_robots:
            return None

        if self.robots_positions_expected is None:
            self.robots_positions_expected = np.array(robots_positions, copy=True)

        # 若需要宏观优化
        if self.goalFlag and not self.StopFlag:
            if self.flag == 1:
                self.StopFlag = 1
                self.stop_flag = True
                return None
            self.robots_positions_expected = np.array(robots_positions, copy=True)
            rpe = self.robots_positions_expected
            # 检查是否需要重新估计当前 GMM
            if len(self.GMM) > 0 and self.step > 0:
                curr = self.GMM[self.step - 1] if self.step > 0 else [self.current_means, self.current_covs, self.current_weights]
                if len(curr) >= 3 and len(curr[0]) > 0:
                    try:
                        scores = self._gmm_score_samples(curr[0], curr[1], curr[2], rpe)
                        if np.min(scores) < np.log(1e-4):
                            self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                                self.conbinedmeans_list, self.conbinedcovs_list, rpe
                            )
                        else:
                            self.current_means, self.current_covs, self.current_weights = curr[0], curr[1], curr[2]
                    except Exception:
                        self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                            self.conbinedmeans_list, self.conbinedcovs_list, rpe
                        )
                else:
                    self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                        self.conbinedmeans_list, self.conbinedcovs_list, rpe
                    )
            else:
                self.current_means, self.current_covs, self.current_weights = control_law_3D.estimate_swarm_GMM_3D(
                    self.conbinedmeans_list, self.conbinedcovs_list, rpe
                )

            if not self.use_gmm_trajectory_slp:
                self.GMM = [[self.fmeans, self.fcovs, self.fweights]]
                self.WStack = [np.eye(1)]
                self.goalFlag = 0
                self.optimization_k += 1
                self.step = 0
            else:
                current_goal = [self.current_means, self.current_covs, self.current_weights]
                if self.step < len(self.GMM) and len(self.GMM) > 0:
                    prev = self.GMM[self.step - 1] if self.step > 0 else [self.current_means, self.current_covs, self.current_weights]
                    if len(prev) >= 3:
                        current_goal = prev

                (
                    self.goal_means,
                    self.goal_covs,
                    self.goal_weights,
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    TransferMatrix,
                    self.flag,
                ) = Planning_3D.Optimization_SLP(
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    self.fmeans,
                    self.fcovs,
                    self.fweights,
                    self.conbinedmeans_list,
                    self.conbinedcovs_list,
                    self.esdf_map,
                    self.alpha,
                    current_goal[0],
                    current_goal[1],
                    current_goal[2],
                    self.Graph_GC,
                    self.Wasserstein_table,
                    self.Node_PDF_table,
                )
                self.GMM, self.WStack = Planning_3D.interpGMM_PRM(
                    self.current_means,
                    self.current_covs,
                    self.current_weights,
                    self.goal_means,
                    self.goal_covs,
                    self.goal_weights,
                    TransferMatrix,
                    self.flag,
                )
                self.goalFlag = 0
                self.optimization_k += 1
                self.step = 0

        if len(self.GMM) == 0:
            return None
        if self.step >= len(self.GMM):
            self.goalFlag = 1
            self.step = len(self.GMM) - 1
        next_m, next_c, next_w = self.GMM[self.step]
        next_means = [list(m) for m in next_m]
        next_covs = [np.asarray(c).reshape(3, 3) if np.size(c) == 9 else c for c in next_c]
        next_weights = list(next_w)

        # APF 一步
        self.robots_positions_expected, robots_positions_list, _, _, _ = control_law_3D.APF(
            next_means,
            next_covs,
            next_weights,
            self.robots_positions_expected,
            self.esdf_map,
            MaxNumTry=self.max_apf_try,
        )

        trajectories = []
        for i in range(self.num_robots):
            traj = np.array([pos[i] for pos in robots_positions_list])
            trajectories.append(traj)

        if self.step == len(self.GMM) - 1:
            self.goalFlag = 1
        self.step += 1
        return trajectories, self.stop_flag
