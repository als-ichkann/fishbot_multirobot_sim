import numpy as np
from matplotlib.patches import Polygon
import scipy.stats as stats
from scipy.optimize import fsolve
import trimesh

def sd_3d(p, oi):
    """
    计算三维空间中点到凸障碍物的符号距离（SDF）
    
    参数：
        point (np.array): 待测点的坐标 [x, y, z]
        obstacle_mesh (trimesh.Trimesh): 表示障碍物的凸多面体网格
    
    返回：
        signed_distance (float): 带符号的距离（内部为负，外部为正）
        nearest_point (np.array): 障碍物表面上的最近点 [x, y, z]
    """
    # 将输入转换为numpy数组
    p = np.asarray(p, dtype=np.float64)
    
    # 使用trimesh的最近点查询
    closest, distance, _ = trimesh.proximity.closest_point(oi, [p])
    nearest_point = closest[0]
    
    # 判断点是否在障碍物内部
    is_inside = oi.contains([p])[0]
    
    # 符号距离规则：内部为负，外部为正
    # ！！输出的distance,nearest_point均为numpy数组，需要转换成标量和列表使用
    signed_distance = -distance.item() if is_inside else distance.item()
    nearest_point = nearest_point.tolist()
    return signed_distance, nearest_point


def normal_vector_SDF_3d(p, oi):
    
    """
    计算三维空间中点的SDF法向量（指向障碍物外部或内部）
    
    参数：
        p (np.array): 待测点的坐标 [x, y, z]
        obstacle_mesh (trimesh.Trimesh): 表示障碍物的凸多面体网格
    
    返回：
        normal (np.array): 单位法向量 [nx, ny, nz]
    """
    p = np.asarray(p).squeeze().astype(np.float64)
    signed_distance, nearest_point = sd_3d(p, oi)
    vector = p - nearest_point
    distance = np.linalg.norm(vector)
    
    if distance > 1e-6:
        # 常规点：使用向量方向归一化
        normal = (vector / distance) * np.sign(signed_distance)
    else:
        # 表面点：直接查询面的法向量
        query = trimesh.proximity.ProximityQuery(oi)
        closest_points, distances, face_indices = query.on_surface([p])
        
        if len(face_indices) > 0:
            # 获取面法向量并确保方向正确
            face_normal = oi.face_normals[face_indices[0]]
            normal = face_normal * np.sign(signed_distance)
            normal_norm = np.linalg.norm(normal)
            
            if normal_norm < 1e-6:
                # 法向量为零向量时的容错处理
                normal = np.array([1.0, 0.0, 0.0])
            else:
                normal = normal / normal_norm
        else:
            # 无法找到面时的默认法向量
            normal = np.array([1.0, 0.0, 0.0])
    
    return normal

def CVaR(gj, covj, obstacle_vertices, alpha):  # compute CVaR for all obstacles
    def percentile_of_point(cdf_func, p, mu, sigma):
        # Solving the inverse function of the cumulative probability of a univariate normal distribution
        def func(x):
            return cdf_func(x, mu, sigma) - p

        result = fsolve(func, mu)
        return result[0]
    def univariate_normal_pdf(x, mean, variance):
        coefficient = 1 / np.sqrt(2 * np.pi * variance)
        exponent = -0.5 * ((x - mean) ** 2 / variance)
        return coefficient * np.exp(exponent)

    maximum = float('-inf')
    for oi in obstacle_vertices:
        covj = np.array(covj)
        mean, _ = sd_3d(gj, oi)           # use Taylor expansion: sd(Xj,oi)=sd(muj,oi)+[grad(sd(Xj,oi))|Xj=muj]*(Xj-muj),so sd(Xj,oi)~N(sd(muj,oi),(nj).T*sigmaj*nj)
        mean = - mean
        nj = normal_vector_SDF_3d(gj, oi)
        sigma = np.dot(np.dot(nj, covj), np.transpose(nj))
        cdf_func = stats.norm.cdf
        percentile_point = percentile_of_point(cdf_func, 1 - alpha, mean, np.sqrt(sigma))
        pdf_value = univariate_normal_pdf(percentile_point, mean, sigma)
        result = mean + sigma * pdf_value / alpha
        if result > maximum:
            maximum = result

    return maximum

def CVaR_for_single_obstacle(gj, covj, oi, alpha):  # compute CVaR for a single obstacle
    def percentile_of_point(cdf_func, p, mu, sigma):
        # Solving the inverse function of the cumulative probability of a univariate normal distribution
        def func(x):
            return cdf_func(x, mu, sigma) - p

        result = fsolve(func, mu)
        return result[0]
    def univariate_normal_pdf(x, mean, variance):
        coefficient = 1 / np.sqrt(2 * np.pi * variance)
        exponent = -0.5 * ((x - mean) ** 2 / variance)
        return coefficient * np.exp(exponent)

    covj = np.array(covj)
    mean, _ = sd_3d(gj, oi)
    mean = - mean
    nj = normal_vector_SDF_3d(gj, oi)
    sigma = np.dot(np.dot(nj, covj), np.transpose(nj))
    cdf_func = stats.norm.cdf
    percentile_point = percentile_of_point(cdf_func, 1 - alpha, mean, np.sqrt(sigma))
    VaR = percentile_point
    pdf_value = univariate_normal_pdf(percentile_point, mean, sigma)
    CVaR = mean + sigma * pdf_value / alpha

    return mean, sigma, VaR, CVaR

def CVaR_for_single_obstacle_random(gj, oi, alpha, random): 
    '''
    def percentile_of_point(cdf_func, p, mu, sigma):
        
        def func(x):
            return cdf_func(x, mu, sigma) - p

        result = fsolve(func, mu)    
        return result[0]
    def univariate_normal_pdf(x, mean, variance):  
        coefficient = 1 / np.sqrt(2 * np.pi * variance)
        exponent = -0.5 * ((x - mean) ** 2 / variance)
        return coefficient * np.exp(exponent)
    '''
    mean, _ = sd_3d (gj, oi)      
    while mean == 0:                    #如果距离为0，增加随机项
        random_mean = np.array(gj) + random
        mean, _ = sd_3d(random_mean.tolist(), oi)
    mean = - mean
    nj = normal_vector_SDF_3d(gj, oi)
    Tau = 10 ** mean                   #风险系数，当mean越小，tau越大
    return nj, Tau
