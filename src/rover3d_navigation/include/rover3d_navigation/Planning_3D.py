import numpy as np
import scipy.linalg
import scipy.sparse
from scipy.stats import norm
try:
    from . import control_law_3D
except ImportError:
    import control_law_3D
#from scipy.optimize import linprog
#from shapely.geometry import Polygon, MultiPoint, LineString
import math
#from scipy.stats import multivariate_normal
import pandas as pd
import time
from qpsolvers import solve_qp
import networkx as nx
from trimesh.ray import ray_pyembree
#import trimesh
# ESDF: use esdf_map parameter (EsdfMapAdapter) with get_esdf, compute_gradient, is_collision_line_segment, resolution


def _find_mean_index(means_list, mean):
    """容差匹配：在 means_list 中查找与 mean 最接近的索引，避免 list.index 的浮点精度问题。"""
    arr = np.asarray(mean).flatten()[:3]
    for i, m in enumerate(means_list):
        if np.allclose(np.asarray(m).flatten()[:3], arr):
            return i
    raise ValueError(f"Mean {list(arr)} not found in means_list (len={len(means_list)})")


def shortest_path(Graph):
    all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(Graph), weight='weight')                   #计算所有节点对的最短路径
    all_pairs_shortest_path_length = dict(nx.all_pairs_dijkstra_path_length(Graph), weight='weight')     #计算所有节点对的最短路径长度
    path_existence = {}     
    path_lengths = {}
    for node in Graph.nodes():
        path_existence[node] = {}
        path_lengths[node] = {}
        for target in Graph.nodes():
            if node != target:
                if target in all_pairs_shortest_path[node]:
                    path = all_pairs_shortest_path[node][target]
                    weighted_length = sum(Graph[u][v]['weight'] for u, v in zip(path[:-1], path[1:]))
                    path_lengths[node][target] = weighted_length
                    path_existence[node][target] = True
                else:
                    path_lengths[node][target] = float('nan')
                    path_existence[node][target] = False
            else:
                path_lengths[node][target] = 0
                path_existence[node][target] = True
    return path_existence, path_lengths

def check_3d_line_obstacle(start_point, end_point, obstacles):
    """
    三维线段与多面体障碍物碰撞检测
    :param start_point: 线段起点 [x, y, z]
    :param end_point: 线段终点 [x, y, z]
    :param obstacles: 障碍物列表，每个元素为trimesh.Trimesh对象
    :return: 是否碰撞 (True/False)
    """
    # create direction vector
    direction = np.array(end_point) - np.array(start_point)
    norm = np.linalg.norm(direction)
    if norm < 1e-6:  # 零长度线段
        return False
    
    # 转换为trimesh射线格式
    ray_origins = np.array([start_point])
    ray_directions = np.array([direction / norm])  # 单位化
    
    # 遍历所有障碍物
    for obstacle in obstacles:
        # 获取障碍物的边界表示
        mesh = obstacle
        
        # 进行射线相交检测
        intersector = ray_pyembree.RayMeshIntersector(mesh)
        locations, _, _ = intersector.intersects_location(
            ray_origins=ray_origins,
            ray_directions=ray_directions)
        
        # 检查交点是否在线段范围内
        if len(locations) > 0:
            for loc in locations:
                # 计算参数t (0 <= t <= 1)
                vec = loc - start_point
                t = np.dot(vec, direction) / np.dot(direction, direction)
                if 0 <= t <= 1:
                    return True
                    
    return False
def notgreedy_genPathTable(current_means, current_covs, current_weights, fmeans, fcovs, fweights, conbinedmeans_list,
                           conbinedcovs_list, esdf_map , Graph_GC, Wasserstein_table):
    
    LinearConnectFlag = 1
    delDist = 0.05
    W_tf = 3
    NodePairTable = []
    '''
    for i in range(len(current_means)):
        current_mu = current_means[i]                                     # 当前GMM的各高斯组分的均值
        current_sigma = current_covs[i]                                   # 当前GMM的各高斯组分的方差
        current_mu_i = _find_mean_index(conbinedmeans_list, current_mu)               # 当前GMM的各高斯组分对应总的离散化节点列表索引
        for n in range(len(conbinedmeans_list)):
            node_mu = conbinedmeans_list[n]                               # 遍历离散化节点列表的每个节点      
            node_sigma = conbinedcovs_list[n]
            d = Wasserstein_table[current_mu_i, n]                        # 读取wasserstein_table中一对索引之间的ws距离
            if d >= np.sqrt(0) and d <= np.sqrt(4):                       # 筛选出ws距离满足大于0小于4的节点
                LinearConnectFlag = 1
                point1 = ([current_mu[0],current_mu[1],current_mu[2]])
                point2 = ([node_mu[0], node_mu[1], node_mu[2]])
                if esdf_map.is_collision_line_segment(point1, point2):
                    LinearConnectFlag = 0                                 # 如果两个坐标相连与障碍物相交，说明这条直连路径不可行，LinearConnectFlag=0
                if LinearConnectFlag == 1:
                    if d < 1e-5:                                          # 如果距离很近，直接置为0
                        Lagrangian = 0
                    else:
                        Dist_sq = math.ceil(d / delDist) * (delDist ** 2) # 否则，将距离用标准化单位delDist表示
                        Lagrangian = Dist_sq 
                    
                    for m in range(len(conbinedmeans_list)):              # 遍历离散化节点列表
                        node_mu_m = conbinedmeans_list[m] 
                        node_sigma_m = conbinedcovs_list[m]
                        d_nm = Wasserstein_table[n, m]
                        if d_nm >= np.sqrt(0) and d_nm <= np.sqrt(4):
                            LinearConnectFlag = 1
                            # x1 = [node_mu_m[0], node_mu[0]]
                            # y1 = [node_mu_m[1], node_mu[1]]
                            point3 = np.clip([node_mu_m[0],node_mu_m[1],node_mu_m[2]])
                            if esdf_map.is_collision_line_segment(point2, point3):
                                LinearConnectFlag = 0
                            if LinearConnectFlag == 1:
                                if d_nm < 1e-5:
                                    Lagrangian2 = 0
                                else:
                                    Lagrangian2 = math.ceil(d_nm / delDist) * (delDist ** 2)
                                table_line = [i, n, m, 0, Lagrangian, Lagrangian2]             # 每条路径为从当前GMM的第i个组分出发，经过第n个节点，到达第m个节点
                                table_lines = [table_line[:] for _ in range(len(fmeans))]      # 创建len（fmeans）条路径副本
                                for ii in range(len(fmeans)): 
                                    table_lines[ii][3] = ii                                    # 为每一个路径副本添加终点索引
                                NodePairTable.extend(table_lines)                              # 将这些路径添加到总的路径表里面去
                                '''
    for i in range(len(current_means)):
        current_mu = current_means[i]
        current_sigma = current_covs[i]
        current_mu_i = _find_mean_index(conbinedmeans_list, current_mu)
        print(f"\n[Level 1] 当前GMM组分 i={i}, mu={current_mu}, index={current_mu_i}")
        for n in range(len(conbinedmeans_list)):
            node_mu = conbinedmeans_list[n]
            node_sigma = conbinedcovs_list[n]
            d = Wasserstein_table[current_mu_i, n]

            if d >= np.sqrt(0) and d <= np.sqrt(4):
                LinearConnectFlag = 1
                point1 = [current_mu[0], current_mu[1], current_mu[2]]
                point2 = [node_mu[0], node_mu[1], node_mu[2]]

                if esdf_map.is_collision_line_segment(point1, point2):
                    LinearConnectFlag = 0
                if LinearConnectFlag == 1:
                    if d < 1e-5:
                        Lagrangian = 0
                    else:
                        Dist_sq = math.ceil(d / delDist) * (delDist ** 2)
                        Lagrangian = Dist_sq

                    for m in range(len(conbinedmeans_list)):
                        node_mu_m = conbinedmeans_list[m]
                        node_sigma_m = conbinedcovs_list[m]
                        d_nm = Wasserstein_table[n, m]

                        if d_nm >= np.sqrt(0) and d_nm <= np.sqrt(4):
                            LinearConnectFlag = 1
                            point3 = np.array([node_mu_m[0], node_mu_m[1], node_mu_m[2]])
                            if esdf_map.is_collision_line_segment(point2, point3):
                                LinearConnectFlag = 0

                            if LinearConnectFlag == 1:
                                if d_nm < 1e-5:
                                    Lagrangian2 = 0
                                else:
                                    Lagrangian2 = math.ceil(d_nm / delDist) * (delDist ** 2)
                                
                                table_line = [i, n, m, 0, Lagrangian, Lagrangian2]
                                table_lines = [table_line[:] for _ in range(len(fmeans))]
                                for ii in range(len(fmeans)):
                                    table_lines[ii][3] = ii
                                NodePairTable.extend(table_lines)
                                print(f"[Add] i={i}, n={n}, m={m}, d={d:.3f}, d_nm={d_nm:.3f}, "
                                    f"L1={Lagrangian}, L2={Lagrangian2}, total added={len(NodePairTable)}")
    path_table = np.zeros((len(NodePairTable), 8))                                             # 初始化八列的路径表
    path_table[:, :6] = np.array(NodePairTable)                                                # 前六列添加原始数据
    numPathTable = path_table.shape[0]                                                         # 路径总数为len（current_means)*(len(conbinedmean_list)^2)*len(fmeans)
    subTable = path_table[:, 2:4]                                                              # 提取子表，第三、四列，即m和ii

    df = pd.DataFrame(subTable)   
    unique_df = df.drop_duplicates()                                                                            # 去重
    unique_df_sorted = unique_df.sort_values(by=unique_df.columns.tolist()).reset_index(drop=True)              # 排序
    PathTable_Unique = unique_df_sorted.to_numpy()                                                              # 转换为numpy数组
    #unique_indices = unique_df_sorted.index
    #subTable_df = pd.DataFrame(subTable)
    #_, idx_Table = np.unique(subTable, axis=0, return_inverse=True)  # The result is temporarily incorrect but not important
    idx_Table_Unique = np.array([np.where(np.all(PathTable_Unique == row, axis=1))[0][0] for row in subTable])  # 为原始路径数据中的每个节点对 (m, ii) 找到在唯一表 PathTable_Unique 中的索引。

    numPathTable_Unique = PathTable_Unique.shape[0]
    dist_n_j_Unique = np.zeros(numPathTable_Unique)
    indexList_n = PathTable_Unique[:, 0]         
    indexList_j = PathTable_Unique[:, 1]
    all_pairs_shortest_path_length = Graph_GC
    for m in range(numPathTable_Unique):
        n = int(indexList_n[m])
        j = int(indexList_j[m])
        inner = all_pairs_shortest_path_length.get(n, {})
        dist_n_j_Unique[m] = inner.get(j, float('nan'))  
    dist_n_j = dist_n_j_Unique[idx_Table_Unique]      
    path_table[:, 6] = dist_n_j                                                       #利用dijkstra算法求出最后一步有向图，从第n个节点到第j个终点均值位置的最短距离
    path_table[:, 7] = path_table[:, 4] + path_table[:, 5] + W_tf * path_table[:, 6]  #算出这条路径的总距离，w_tf为常数
    path_table = path_table[~np.isnan(dist_n_j), :]
    return path_table


def Optimization_SLP(current_means, current_covs, current_weights, fmeans, fcovs, fweights, conbinedmeans_list,
                     conbinedcovs_list, esdf_map, alpha, current_goal_means,
                     current_goal_covs, current_goal_weights, Graph_GC, Wasserstein_table, Node_PDF_table):
    
    print("[0] 开始 Optimization_SLP")
    mu = current_means  # current gmm parameters
    sigma = current_covs
    weights = current_weights
    flag = 0
    epsilon = 0.2
    min_weight = 0.002
    tau = 1e-3
    gamma = 1
    aa = 1 / np.e ** 3
    MaxPDF = 1e-1
    curr_GC_index = []

    print(current_means)
    print(current_covs)
    print(current_weights)
    #将当前GMM的每个分量均值（mu[i]）与全局组合列表（combined_means_list）中的对应位置匹配
    for i in range(len(current_means)):
        j = _find_mean_index(conbinedmeans_list, mu[i])
        curr_GC_index.append(j)
    print("[1] 匹配 GC 索引完成")
    # ESDF风险参数初始化（单障碍物模式）
    v = np.array([esdf_map.get_esdf(conbinedmeans_list[idx]) for idx in curr_GC_index])  # 形状(numnode,)
    Smu = v  # 均值参数，直接用一维数组即可

    # 正确计算每个分量的方差参数
    Ssig = np.array([
        np.dot(
            grad.T, np.dot(current_covs[i], grad)
        ) if (grad := esdf_map.compute_gradient(conbinedmeans_list[curr_GC_index[i]])) is not None else 0
        for i in range(len(current_means))
    ])
    print("[2] 构建 Smu/Ssig 完成")

    '''
    # 二分法计算VaR，增加最大迭代次数防止死循环
    max_iter = 100
    iter_count = 0
    VatR = 0.0
    vmax, vmin = np.max(v), np.min(v)
    while iter_count < max_iter:
        VatR = 0.5 * (vmax + vmin)
        Ualpha = 1 - norm.cdf(VatR, loc=Smu, scale=np.sqrt(Ssig))
        alpha0 = weights @ Ualpha
        if abs(alpha0 - alpha) < 1e-10:
            break
        if alpha0 < alpha:
            vmin = VatR
        else:
            vmax = VatR
        iter_count += 1
    else:
        print("Warning: VaR bisection did not converge within max_iter.")

    # 计算CVaR
    alphaj = 1 - norm.cdf(VatR, loc=Smu, scale=np.sqrt(Ssig))
    pdf_value = norm.pdf(norm.ppf(1 - alphaj))
    gra = (alphaj * Smu + pdf_value * np.sqrt(Ssig)) / alpha
    CVaR_curr = weights @ gra

    if CVaR_curr > epsilon:
        print('current_gmm_est is larger than epsilon')
        current_means, current_covs, current_weights = current_goal_means, current_goal_covs, current_goal_weights

    # 安全检查结束，评估当前GMM与目标的接近程度,若 WG_sq 在 [0,1] 范围内，说明当前GMM已足够接近目标，无需优化
    '''
    WG_sq, W, _ = control_law_3D.calWGMetric_speedUp(current_means, current_covs, current_weights, fmeans, fcovs, fweights)
    print(f"[2] Wasserstein distance squared: {WG_sq:.4f}")
    if 0 <= WG_sq and WG_sq <= 1:
        Next_means = fmeans
        Next_covs = fcovs
        Next_weights = fweights
        TransferMatrix = W
        flag = 1
        return Next_means, Next_covs, Next_weights, current_means, current_covs, current_weights, TransferMatrix, flag
    # 过滤无效的GMM分量,重新归一化权重，确保总和为1。
    weights = current_weights
    indexList = np.where(np.array(weights) >= min_weight)[0]
    if len(indexList) < len(current_means):
        current_means = current_means[indexList]
        current_covs = current_covs[indexList]
        weights = weights[indexList]
        current_weights = weights / sum(weights)
    print("[3] 路径表开始")
    # 生成非贪婪路径表
    start = time.time()
    path_table = notgreedy_genPathTable(current_means, current_covs, current_weights, fmeans, fcovs, fweights,
                                        conbinedmeans_list, conbinedcovs_list, esdf_map, Graph_GC, Wasserstein_table)
    print(f"[4] 路径表完成，耗时 {time.time() - start:.2f}s")
    #print(f"路径表：{path_table}")

    # 路径表为空（图中无可达路径）时，直接以目标为下一步并返回
    if path_table.shape[0] == 0:
        print("[4] 警告：路径表为空（无可达路径），回退为直达目标")
        _, W, _ = control_law_3D.calWGMetric_speedUp(current_means, current_covs, current_weights, fmeans, fcovs, fweights)
        return fmeans, fcovs, fweights, current_means, current_covs, current_weights, W, 1

    # 节点索引处理
    index_next_gc = np.unique(path_table[:, 1]).astype(int)
    index_k2_gc = np.unique(path_table[:, 2]).astype(int)
    num_next_gc = len(index_next_gc)
    num_k2_gc = len(index_k2_gc)


    # ESDF风险参数提取
    mean = np.array([esdf_map.get_esdf(conbinedmeans_list[idx]) for idx in index_next_gc])  # 修正拼写错误
    meank2 = np.array([esdf_map.get_esdf(conbinedmeans_list[idx]) for idx in index_k2_gc])
    
    sigma = np.array([
        (np.dot(grad.T, np.dot(conbinedcovs_list[idx], grad))  # 使用combinedcovs_list中的协方差矩阵
         if (grad := esdf_map.compute_gradient(conbinedmeans_list[idx])) is not None 
         else 0)
        for idx in index_next_gc
    ])
    
    sigmak2 = np.array([
        (np.dot(grad.T, np.dot(conbinedcovs_list[idx], grad))  # 同上修正
         if (grad := esdf_map.compute_gradient(conbinedmeans_list[idx])) is not None 
         else 0)
        for idx in index_k2_gc
    ])
    # 计算中间节点风险v 
    z_score = norm.ppf(1 - alpha)  # 根据置信水平获取分位数
    v = np.array([
        mean[i] - z_score * np.sqrt(sigma[i])  # VaR计算：均值 - z分位数*标准差
        for i in range(len(index_next_gc))
    ])
    # 计算目标节点风险vk2 
    vk2 = np.array([
        meank2[i] - z_score * np.sqrt(sigmak2[i]) 
        for i in range(len(index_k2_gc))
    ])

    numVariable = path_table.shape[0]           # 变量数 = 路径表行数
    f = path_table[:, 7]                        # 目标函数系数，即每条路径的总运输成本
    cost = []                                   # 初始化成本存储

    # =========begin SLP===============

    iter = 0
    soludiff = []
    ConvFlag = 0
    ConvCounter = 0
    ConvThredhold = 0.003


    print("[5] CVaR 迭代开始")
    while ConvFlag != True:
        Wa = np.zeros(num_next_gc)                          # Wa为初始化中间节点（n）的权重
        Wak2 = np.zeros(num_k2_gc)                          # Wak2为初始化目标节点（m）的权重
        # 若是第一步迭代
        if iter == 0:                                   
            Path_curr = np.zeros(numVariable)               # 初始化路径光标数组，标记每条路径的当前节点索引
            for i in range(len(current_means)):
                j = _find_mean_index(conbinedmeans_list, current_means[i])
                Wa[j == index_next_gc] = weights[i]         # 那些当前步与中间步相重合的节点，其权重初始化为当前步权重
                Wak2[j == index_k2_gc] = weights[i]         # 那些当前步与目标步相重合的节点，其权重初始化为当前步权重
                indices = np.where(path_table[:, 0] == i)   # 找到所有以当前分量i为起点的路径
                Path_curr[indices] = j                      # 将这些路径的光标标记为j
            PIa = np.zeros(numVariable)                     # 初始化路径分布概率数组
            PIa_id = np.where((Path_curr == path_table[:, 1]) & (path_table[:, 1] == path_table[:, 2]))[0]  #满足当前节点=中间节点=目标节点，即从当前节点到最终分布的直达路径
            for u in range(len(current_weights)):
                for ii in range(len(fmeans)):
                    idx = len(fmeans) * u + ii
                    if idx < len(PIa_id):
                        PIa[PIa_id[idx]] = weights[u] * fweights[ii]   #初始化这些直达路径的概率
        
        # 其余迭代步，用优化后的权重更新中间分布和规划域内目标分布的权重
        else:
            Wa = Ws
            Wak2 = Wsk2
        #-------ESDF风险参数计算-------
        # 计算中间节点风险参数
        sigma_proj = np.array([
            np.dot(esdf_map.compute_gradient(conbinedmeans_list[idx]).T, 
                np.dot(conbinedcovs_list[idx], 
                        esdf_map.compute_gradient(conbinedmeans_list[idx])))
            if esdf_map.compute_gradient(conbinedmeans_list[idx]) is not None else 0
            for idx in index_next_gc
        ])

        # 计算目标节点风险参数 
        sigma_proj_k2 = np.array([
            np.dot(esdf_map.compute_gradient(conbinedmeans_list[idx]).T,
                np.dot(conbinedcovs_list[idx],
                        esdf_map.compute_gradient(conbinedmeans_list[idx])))
            if esdf_map.compute_gradient(conbinedmeans_list[idx]) is not None else 0
            for idx in index_k2_gc
        ])

        # 风险值迭代计算 
        def compute_risk_params(mean_vals, sigma_vals, weights, alpha, max_iter=100):
            """ESDF风险参数计算核心函数"""
            VatR = 0.0
            v_max, v_min = np.max(mean_vals), np.min(mean_vals)
            
            # 二分法收敛
            for _ in range(max_iter):
                VatR = 0.5 * (v_max + v_min)
                exceed_prob = 1 - norm.cdf(VatR, loc=mean_vals, scale=np.sqrt(sigma_vals))
                total_risk = np.dot(weights, exceed_prob)
                
                if abs(total_risk - alpha) < 1e-14:
                    break
                if total_risk > alpha:
                    v_min = VatR
                else:
                    v_max = VatR
            
            # CVaR计算
            z_alpha = norm.ppf(1 - alpha)
            pdf_val = norm.pdf(z_alpha)
            cvar = (mean_vals - (sigma_vals * pdf_val) / alpha)
            grad_cvar = - (pdf_val/alpha) * (1 / (2 * np.sqrt(sigma_vals + 1e-6)))

            
            return VatR, cvar, grad_cvar
        '''
        # 中间节点风险计算
        VatR, CVaR_a , grad_risk = compute_risk_params(mean, sigma_proj, Wa, alpha)
        # 目标节点风险计算
        VatR_k2, CVaR_a_k2, grad_risk_k2= compute_risk_params(meank2, sigma_proj_k2, Wak2, alpha)
        '''
        # weight bound
        #lb = 0
        #ub = 1

        # 不等式约束矩阵构建
        # -------概率密度约束---------
        A_PDF = np.zeros((len(index_next_gc), numVariable))
        for n in range(numVariable):
            A_PDF[:, n] = Node_PDF_table[index_next_gc, int(path_table[n,1])] * (esdf_map.resolution**3)

        A_PDF_k2 = np.zeros((len(index_k2_gc), numVariable))
        for n in range(numVariable):
            A_PDF_k2[:, n] = Node_PDF_table[index_k2_gc, int(path_table[n,2])] * (esdf_map.resolution**3)
        
        '''
        A_risk = np.zeros((len(index_next_gc), numVariable))
        for n in range(numVariable):
            index = np.where(path_table[:, 1] == index_next_gc[n])[0]
            if len(index) > 0:
                A_risk[:, n] = grad_risk[index] * (esdf_map.resolution**3)
        A_risk_k2 = np.zeros((len(index_k2_gc), numVariable))
        for n in range(numVariable):
            index = np.where(path_table[:, 2] == index_k2_gc[n])[0]
            if len(index) > 0:
                A_risk_k2[:, n] = grad_risk_k2[index] * (esdf_map.resolution**3)
        '''

        A = np.vstack([A_PDF, A_PDF_k2])
        b_PDF1 = np.full((len(index_next_gc),), MaxPDF)
        b_PDF2 = np.full((len(index_k2_gc),), MaxPDF)
        b = np.concatenate([b_PDF1, b_PDF2], axis=0)
        '''
        b_risk1 = np.full((len(index_next_gc),), epsilon) - Wa @ grad_risk
        b_risk2 = np.full((len(index_k2_gc),), epsilon) - Wak2 @ grad_risk_k2
        '''

        b = b.astype(np.float64).flatten()  # Ensure shape (N,) and dtype float
        # ---------等式约束矩阵 ---------
        Aeq = []
        beq = []
        #print(f"current_means: {current_means}")
        #print(f"conbinedmeans_list: {conbinedmeans_list}")
        for n in range(len(current_means)):
            index = np.where((path_table[:, 0]) == n)[0]
            a = np.zeros(numVariable)
            a[index] = 1
            Aeq.append(a)
            beq.append(current_weights[n])
        for n in range(len(fmeans)):
            index = np.where((path_table[:, 3]) == n)[0]
            a = np.zeros(numVariable)
            a[index] = 1
            Aeq.append(a)
            beq.append(fweights[n])
        Aeq = np.array(Aeq)
        beq = np.array(beq)


        # -------定义QP问题（对线性近似域上进行优化--------
        # minimize 0.5 x^T H x + f^T x
        H = tau * np.eye(numVariable)          # 近端稳定项（离上次迭代的解的距离不要太大）
        f_sqp = f - tau * PIa                  # 线性目标项(路径成本最小化)
        # 输入 sanitization，避免 NaN/Inf 导致 SCS 等求解器报 "Error parsing inputs"
        H = np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        f_sqp = np.nan_to_num(f_sqp, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        Aeq = np.nan_to_num(Aeq, nan=0.0, posinf=0.0, neginf=0.0)
        beq = np.nan_to_num(beq, nan=0.0, posinf=0.0, neginf=0.0)
        # 保证连续内存与 float64（部分求解器要求）
        H = np.ascontiguousarray(H, dtype=np.float64)
        f_sqp = np.ascontiguousarray(f_sqp.flatten(), dtype=np.float64)
        A = np.ascontiguousarray(A, dtype=np.float64)
        b = np.ascontiguousarray(b.flatten(), dtype=np.float64)
        Aeq = np.ascontiguousarray(Aeq, dtype=np.float64)
        beq = np.ascontiguousarray(beq.flatten(), dtype=np.float64)

        time1 = time.time()
        solution = solve_qp(
            P=scipy.sparse.csc_matrix(H),
            q=f_sqp,
            G=scipy.sparse.csc_matrix(A),
            h=b,
            A=scipy.sparse.csc_matrix(Aeq) if Aeq.size > 0 else None,
            b=beq if beq.size > 0 else None,
            lb=np.zeros(H.shape[0]),
            ub=np.ones(H.shape[0]),
            solver='mosek',
            verbose=False,
        )
        if solution is None:
            raise RuntimeError(
                "QP 求解失败：mosek 返回 None，问题可能不可行。"
                "请检查约束与目标设置。"
            )
        fval = 0.5 * np.dot(solution.T, np.dot(H, solution)) + np.dot(f_sqp, solution)
        print(f"QP solution: {solution}, fval: {fval}")
        time2 = time.time()
        print(f'solver_qp time {time2-time1} s')
        print("[7] 解 QP 完成")

        #更新路径概率，若连续3次迭代的解变化小于阈值，则认为收敛。
        cost.append(fval)
        soludiff.append(np.linalg.norm(solution - PIa))
        if soludiff[iter] < ConvThredhold:
            ConvCounter = ConvCounter + 1
        else:
            ConvCounter = 0
        if ConvCounter > 2:
            ConvFlag = True
        if iter > 0:
            gamma = gamma * (1 - aa * gamma)   # 动量法衰减因子
        PIa = PIa + gamma * (solution - PIa)   # 动量法迭代新的路径概率

        # 更新中间节点和目标节点的权重
        Wss = np.zeros(num_next_gc)
        for n in range(index_next_gc.shape[0]):
            index = index_next_gc[n]
            indexList = np.where(path_table[:, 1] == index)
            Wss[n] = np.sum(PIa[indexList])    # 对每个中间节点n，把所有通向它的路径的流量相加，得到该节点的总权重
        Ws = Wss
        Wsk2 = np.zeros(num_k2_gc)
        for n in range(index_k2_gc.shape[0]):
            index = index_k2_gc[n]
            indexList = np.where(path_table[:, 2] == index)
            Wsk2[n] = np.sum(PIa[indexList])   # 对每个目标节点n，把所有通向它的路径的流量相加，得到该节点的总权重
        iter = iter + 1


    #  ======SLP结束，求得最终的solution（每条路径的流量），构建转移矩阵TransferMatrix======
    W = solution
    mu = []
    Sigma = []
    weight = []
    TransferMatrix = np.zeros((len(current_means), len(conbinedmeans_list)))   # 初始化转移矩阵，行为当前GMM每个分量，列为全局组合列表的每个分量，表示从第i个分量到第j个分量的转移权重
    for n in range(len(index_next_gc)):
        index = index_next_gc[n]                                               # 第n个中间节点对应的全局索引
        indexList = np.where(path_table[:, 1] == index)[0]                     # 找到所有通向该中间节点的路径索引
        for j in range(len(indexList)):
            TransferMatrix[path_table[indexList[j], 0].astype(np.int32), index] = TransferMatrix[path_table[
                indexList[j], 0].astype(np.int32), index] + W[indexList[j]]

    indexlist2 = np.where(TransferMatrix < min_weight)
    ppp = np.sum(TransferMatrix, axis=1)
    TransferMatrix[indexlist2] = 0
    TransferMatrix = (TransferMatrix.T / np.sum(TransferMatrix, axis=1)).T * ppp[:, np.newaxis]
    TransferMatrix[np.isinf(TransferMatrix) | np.isnan(TransferMatrix)] = 0
    qqq = np.sum(TransferMatrix, axis=0)                                      # 按列求和，得到每个“目标节点列”的汇总权重（就是将来 GMM 的 weight 值来源）
    # 过滤低权重路径并归一化
    indexlist3 = np.where(qqq > 0)[0]
    print(f"[6] TransferMatrix 计算完成，非零列数: {len(indexlist3)}")
    print("TransferMatrix 每个非零元素信息：")
    for i in range(TransferMatrix.shape[0]):
        for j in range(TransferMatrix.shape[1]):
            if abs(TransferMatrix[i, j]) > 1e-9:
                print(f"({i}, {j}) = {TransferMatrix[i, j]:.6f}")
    for n in range(len(index_next_gc)):
        index = index_next_gc[n]
        if index in indexlist3.tolist():
            mu.append(conbinedmeans_list[index])
            Sigma.append(conbinedcovs_list[index])
            weight.append(qqq[index])

    # 提取最终GMM参数
    total_weights = sum(weight)
    weight = [weight_single / total_weights for weight_single in weight]
    next_mu, next_sigma, next_weight = mu, Sigma, weight
    print(f"[7] 最终 GMM 均值: {next_mu}, 方差: {next_sigma}, 权重: {next_weight}")
    return next_mu, next_sigma, next_weight, current_means, current_covs, current_weights, TransferMatrix, flag


def interpGMM_PRM(means1, covs1, weights1, means2, covs2, weights2, TransferMatrix, flag):
    delDist = 0.2
    dimW = 3
    WG_sq, W, _ = control_law_3D.calWGMetric_speedUp(means1, covs1, weights1, means2, covs2, weights2)
    d = np.sqrt(WG_sq)
    numPoint = math.ceil(d / delDist)
    W = TransferMatrix
    if flag == 1:
        W = W.reshape((len(means1), len(means2)), order='F')
        W = np.delete(W, np.where(np.all(W == 0, axis=0))[0], axis=1)
    GMM = []
    WStack = []
    if numPoint <= 1:
        GMM.append([means2, covs2, weights2])
        WStack.append(W)
        return GMM, WStack
    t = np.linspace(0, 1, numPoint + 1)
    t = t[1:]
    W = np.delete(W, np.where(np.all(W == 0, axis=0))[0], axis=1)
    idxListW = np.where(W.flatten(order='F') > 0)[0]
    numComponent_p = idxListW.shape[0]
    W_0 = np.zeros((len(means1), numComponent_p))
    W_1 = np.zeros((numComponent_p, len(means2)))
    Weight = np.zeros(numComponent_p)
    for k in range(numComponent_p):
        n, m = np.unravel_index(idxListW[k], (len(means1), len(means2)), order='F')
        Weight[k] = W[n, m]
        W_0[n, k] = W[n, m]
        W_1[k, m] = W[n, m]
    W_0_sum = np.sum(W_0, axis=1, keepdims=True)
    W_0 = W_0 / W_0_sum
    W_1_sum = np.sum(W_1, axis=1, keepdims=True)
    W_1 = W_1 / W_1_sum

    for i in range(t.shape[0] - 1):
        t_i = t[i]
        Mu = np.zeros((numComponent_p, dimW))
        Sigma = np.zeros((numComponent_p, dimW, dimW))
        for k in range(numComponent_p):
            n, m = np.unravel_index(idxListW[k], (len(means1), len(means2)), order='F')
            sigma_p0_sqrt = scipy.linalg.sqrtm(np.array(covs1[n]))
            Mu[k, :] = (1 - t_i) * np.array(means1[n]) + t_i * np.array(means2[m])
            temp = scipy.linalg.sqrtm(sigma_p0_sqrt @ np.array(covs2[m]) @ sigma_p0_sqrt)
            Sigma[k, :, :] = np.linalg.inv(sigma_p0_sqrt) @ ((1 - t_i) * np.array(covs1[n]) + t_i * temp) @ (
                    (1 - t_i) * np.array(covs1[n]) + t_i * temp) @ np.linalg.inv(sigma_p0_sqrt)
            Sigma[k, 0, 1] = Sigma[k, 1, 0]
        GMM.append([Mu.tolist(), Sigma.tolist(), Weight.tolist()])
        if i == 1:
            WStack.append(W_0)
        else:
            WStack.append(np.eye(numComponent_p))
    GMM.append([means2, covs2, weights2])
    print(f"[8] 插值 GMM 生成完成: {len(GMM)} 步")
    WStack.append(W_1)
    return GMM, WStack

