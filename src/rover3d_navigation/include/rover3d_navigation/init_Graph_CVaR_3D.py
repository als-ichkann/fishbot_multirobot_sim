import numpy as np
#from scipy.sparse import csr_matrix
try:
    from rover3d_navigation import control_law_3D
except ImportError:
    import control_law_3D
import math
from scipy.stats import multivariate_normal
import scipy
from shapely.geometry import Point, Polygon
from tqdm import tqdm
#from scipy.optimize import minimize
#import CVaR_SDF_constraint_3D as constraint

'''
def genLabelPointMatrix(numGridX, numGridY, numGridZ,  gridX, gridY, gridZ, obstacle_vertices):
    labPoint = np.zeros((numGridY, numGridX))
    for n in range(len(obstacle_vertices)):
        obstacle = obstacle_vertices[n]
        for j in range(numGridY):
            for i in range(numGridX):
                if labPoint[j, i] == 0:
                    obstacle_polygon = Polygon(obstacle)            # plot the vertices of the obstacle
                    point = Point(gridX[j, i], gridY[j, i])         # get x and y
                    point = point.buffer(1e-9)   
                    if point.intersects(obstacle_polygon.boundary):
                        in_polygon = 1
                        on_boundary = 1
                    elif obstacle_polygon.contains(point):
                        in_polygon = 1
                        on_boundary = 0
                    else:
                        in_polygon = 0
                        on_boundary = 0
                    labPoint[j, i] = int(in_polygon) + int(on_boundary)
    indexOut = np.where(labPoint.flatten(order='F') == 0)[0]        # in the free space
    indexOn = np.where(labPoint.flatten(order='F') == 2)[0]         # on the boundary
    indexIn = np.where(labPoint.flatten(order='F') == 1)[0]         # in the obstacle
    indexOcc = np.where(labPoint.flatten(order='F') > 0)[0]         # occupied
    return labPoint, indexOut, indexOn, indexIn, indexOcc   
'''

def interpGC_speedUp(mean1, cov1, mean2, cov2, gridXYZ, delDist):
    # 计算 Wasserstein 距离并确定插值步数
    d = control_law_3D.Wasserstein_distance(mean1, cov1, mean2, cov2)
    mean1, cov1, mean2, cov2 = np.array(mean1), np.array(cov1), np.array(mean2), np.array(cov2)
    numPoint = math.ceil(d / delDist)
    
    # 若无需插值，直接返回终态 PDF
    if numPoint <= 1:
        PDF = multivariate_normal.pdf(gridXYZ, mean=mean2, cov=cov2).astype(np.float32)
        PDF_Vectors = (PDF.reshape(-1, 1) * 0.1**3).astype(np.float32)
        return PDF_Vectors, []
    
    # 初始化预分配数组
    t = np.linspace(0, 1, numPoint + 1)[1:]  # 移除起始点
    N = gridXYZ.shape[0]                     # 网格点数量
    PDF_Vectors = np.zeros((N, numPoint), dtype=np.float32)  # 直接预分配为 (N, numPoint)
    Dist_sq_Vector = np.full(numPoint, delDist**2, dtype=np.float32)  # 预生成距离平方
    
    # 计算协方差插值的公共部分（若 cov1 != cov2）
    if not np.array_equal(cov1, cov2):
        Sigma0_sqrt = scipy.linalg.sqrtm(cov1)
        temp_common = scipy.linalg.sqrtm(Sigma0_sqrt @ cov2 @ Sigma0_sqrt)
    
    for i in range(numPoint):
        t_i = t[i]
        # 插值均值
        mu = (1 - t_i) * mean1 + t_i * mean2
        
        # 插值协方差
        if not np.array_equal(cov1, cov2):
            Sigma = np.linalg.inv(Sigma0_sqrt) @ ((1 - t_i) * cov2 + t_i * temp_common) @ np.linalg.inv(Sigma0_sqrt)
        else:
            Sigma = cov1
        
        # 计算 PDF 并存储为 float32
        PDF = multivariate_normal.pdf(gridXYZ, mean=mu, cov=Sigma).astype(np.float32)
        PDF_Vectors[:, i] = PDF * 0.1**3  # 直接填充列
    
    return PDF_Vectors, Dist_sq_Vector


def init_Graph_GC(conbinedmeans_list, conbinedcovs_list, Wasserstein_table,
                 xa=None, ya=None, za=None, xb=None, yb=None, zb=None):
    """构建 GC 图。
    xa,ya,za,xb,yb,zb: 地图边界，用于插值网格。未提供则使用默认 (0,0,0,20,16,2)。
    """
    print('Start to create the graph...')
    if xa is None:
        xa, ya, za = 0, 0, 0
        xb, yb, zb = 20, 16, 2
    else:
        xa, ya, za = float(xa), float(ya), float(za)
        xb, yb, zb = float(xb), float(yb), float(zb)
    # 粗网格加速：0.25→1.0，网格点从 ~4.7万 降至 ~1千，初始化从数十分钟降至数十秒
    dx, dy, dz = 1.0, 1.0, 1.0
    # alpha_obstacle = 0
    delDist = 0.05
    # epsilon = 0.2 
    # alpha_C = 0.2
    numGridX = int((xb - xa) / dx + 1)
    numGridY = int((yb - ya) / dy + 1)
    numGridZ = int((zb - za) / dz + 1)
    numGrid = numGridX * numGridY * numGridZ
    xGrid = np.linspace(xa, xb, numGridX)
    yGrid = np.linspace(ya, yb, numGridY)
    zGrid = np.linspace(za, zb, numGridZ)

    gridX, gridY, gridZ = np.meshgrid(xGrid, yGrid, zGrid, indexing='ij')
    gridX, gridY, gridZ = np.round(gridX, decimals=1), np.round(gridY, decimals=1),np.round(gridZ,decimals=1)
    gridXYZ = np.column_stack((gridX.flatten(order='F'), gridY.flatten(order='F'),gridZ.flatten(order='F')))

    # labPointPri, indexOutPri, indexOnPri, indexInPri, indexOccPri = genLabelPointMatrix(numGridX, numGridY, gridX, gridY, obstacle_vertices)

    # H = np.zeros((len(conbinedmeans_list), len(conbinedmeans_list)))
    # c = np.max(CVaR.T, axis=0)    # find the maximum CVaR for every node
    D_stack = []
    PDF_Vectors_stack = []
    IndexList_Edges_stack = []
    for i in tqdm(range(len(conbinedmeans_list))):      # tqdm: showing the progress of these codes  # 
        mu_i = conbinedmeans_list[i]
        Sigma_i = conbinedcovs_list[i]
        D = np.zeros((1, len(conbinedmeans_list)))
        PDF_Vectors = np.zeros((numGrid, len(conbinedmeans_list)))
        IndexList_Edges = np.zeros(len(conbinedmeans_list))
        for j in range(len(conbinedmeans_list)):             # find nodes connected to node_i   how to define connected?
            mu_j = conbinedmeans_list[j]   
            Sigma_j = conbinedcovs_list[j]
            d = Wasserstein_table[i, j]                      # 1. Wasserstein distance is lower than 2 and upper than 0
            idx = j * len(conbinedmeans_list) + i           
            if 0 <= d and d <= 2:
                l_sq = math.ceil(d / delDist) * (delDist ** 2)  # distance normalized
                D[:, j] = l_sq
                PDF_vector, _ = interpGC_speedUp(mu_i, Sigma_i, mu_j, Sigma_j, gridXYZ, delDist)
                PDF_vector = np.sum(PDF_vector, axis=1)     #  represent the whole interpolation path probability, from node i to node j
                PDF_Vectors[:, j] = PDF_vector
                IndexList_Edges[j] = idx                    # the edge from node_i to node_j    every edge store an index
        D_stack.append(D)              
        PDF_Vectors = PDF_Vectors[:, IndexList_Edges > 0]   # find those nodes connected to node_i and get its probability vector
        PDF_Vectors_stack.append(PDF_Vectors)
        IndexList_Edges = IndexList_Edges[IndexList_Edges > 0]
        IndexList_Edges_stack.append(IndexList_Edges)
        del IndexList_Edges, PDF_Vectors                    # delete temporary variables
    D = np.array(D_stack).reshape(len(conbinedmeans_list), len(conbinedmeans_list))
    IndexList_Edges = np.hstack(IndexList_Edges_stack)      # combine node_i ' s indexList_Edges for i in numnodes into a line
    PDF_Vectors = np.hstack(PDF_Vectors_stack) 
    GraphA = D
    return GraphA

def init_GC_Nodes(mean_table):

    #初始化高斯节点，包含三个列表，GC_means记录各个节点的均值，GC_covs记录各个节点的方差
    GC_means = []
    GC_covs = []
    for i in range(len(mean_table)):
        mu = mean_table[i]
        Sigma = [[0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
        GC_means.append(mu)
        GC_covs.append(Sigma)

    return GC_means, GC_covs
