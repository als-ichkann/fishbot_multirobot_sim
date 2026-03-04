import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import block_diag
import time
from copy import deepcopy
from qpsolvers import solve_qp, Problem, solve_problem
from .Controller import controller


def get_3d_trajectories(current_points, points_list):
    """
    三维轨迹分割：从离当前点最近的位置截取后续轨迹。
    current_points: (3,) 或 (1,3) 当前坐标
    points_list: list of (N,3) 轨迹
    返回: list of (M,3) 截取后的轨迹
    """
    def find_closest_index(query_point, trajectory):
        distances = cdist(trajectory, np.array([query_point]).reshape(1, -1))
        return int(np.argmin(distances))

    trajectories = []
    for traj in points_list:
        traj_3d = np.array(traj).reshape(-1, 3)
        if traj_3d.size == 0:
            trajectories.append(traj_3d)
            continue
        pt = np.array(current_points).reshape(3)
        closest_idx = find_closest_index(pt, traj_3d)
        trajectories.append(traj_3d[closest_idx:])
    return trajectories


class Agent_3D:
    def __init__(self, name, rp, vmax, vzmax, wmax, controlFrequency):
        self.name = name
        self.initialPosition = rp[0]                      # 初始位置取自路径点列表的第一个元素
        self.rp = rp                                        # 路径点（reference points）列表，可能用于导航或任务规划
        self.radius = 0.3                                   # 机器人半径（明显影响控制效果，控制两机器人最小距离至少为两倍半径）
        self.sensorRange = 3 * self.radius                  # 机器人感知范围
        self.vmax = vmax                                    # 最大移动速度(xy平面内)
        self.vzmax = vzmax                                  # 最大移动速度（z轴）
        self.wmax = wmax                                # 最大角速度
        self.vwork = 0.1                                # 工作状态下的基准速度（可能用于匀速运动）
        self.velocity = np.array([0, 0, 0])             # 当前速度向量（三维空间）
        self.position = self.initialPosition            # 当前位置（初始化为路径起点）
        self.futureState = []                           # 预测或规划的未来状态（可能用于MPC或轨迹预测）                      
        self.acceleration = np.array([0, 0, 0])         # 加速度向量（三维空间）
        self.controlFrequency = controlFrequency        # 控制频率（单位：Hz，影响控制周期）
        self.theta = [0, 0, 0]                          # 朝向角度（列表可能用于记录历史角度）
        self.path = np.array(self.initialPosition, dtype=float).reshape(1, 3)

class MPC_3D:
    def __init__(self, num_robots, N, dt, discrete_points, *args):

        self.discrete_points = np.array(discrete_points).reshape((num_robots, len(discrete_points[0]) * 3))  #原始列表为二维的,现将其改为三维的
        if len(args) > 0:
            self.eng = args[0]
        self.num_robots = num_robots
        self.Ncar = self.num_robots
        self.N = N  # Prediction total steps
        self.n = 9  # number of states
        self.m = 3  # number of control dimensions
        self.NT = 1000
        self.dt = dt
        self.v_max = 0.3  # Maximum speed, simu: 0.3, real: 0.1
        self.vz_max = 0.03
        self.w_max = np.pi / 4  # Maximum angular speed
        self.a_max = 1
        self.jerk = 10
        self.penalty = 10
        self.controlFrequency = 1 / self.dt
        # 支持负坐标，适配仿真工作空间（如 Gazebo 中机器人常在原点附近）
        self.xa0, self.xb0, self.ya0, self.yb0, self.za0, self.zb0 = -20, 20, -20, 20, 0, 20
        self.rate0 = 1
        self.xa, self.xb, self.ya, self.yb, self.za, self.zb = self.xa0 * self.rate0, self.xb0 * self.rate0, self.ya0 * self.rate0, self.yb0 * self.rate0, self.za0 * self.rate0 , self.zb0 * self.rate0
        self.axisLimits = np.array([self.xa, self.xb, self.ya, self.yb, self.za, self.zb])
        self.robot_pose = [None] * self.Ncar
        self.cmd_v = [None] * self.Ncar
        self.rot_pos = [[0, 0] for _ in range(self.Ncar)]
        self.rbt_theta_rad = [0] * self.Ncar
        self.goal_distance = [0] * self.Ncar
        self.v = np.zeros(self.Ncar)
        self.w = np.zeros(self.Ncar)
        self.lastz = []
        #iniPosition = init_robot_positions
        #self.iniPosition_ininitialmap = np.copy(iniPosition)

        self.agents = []
        time_interval = np.zeros((num_robots, int(self.discrete_points.shape[1] / 3 - 1)))
        # time_interval = np.zeros((Ncar,len(mulist)-1))
        # set time interval
        for i in range(self.discrete_points.shape[0]):
            for j in range(0, self.discrete_points.shape[1] - 5, 3):
                time_interval[i, int(j / 3)] = np.ceil(
                    np.linalg.norm(self.discrete_points[i, j:j + 3] - self.discrete_points[i, j + 3:j + 6]) / self.v_max * 4 / 3)
        self.time_interval = np.ceil(np.sum(time_interval, axis=0) / 1)
        self.reshapedPoints = self.discrete_points          # 直接引用已整理好的 (Ncar, T*3) 数组
        for i in range(self.Ncar):
            traj_i   = self.reshapedPoints[i].reshape(-1, 3)         # (T, 3)
            ref_pts  = referencePoints(traj_i,
                                    time_interval[i],             # ← 每台车自己的时间戳序列
                                    self.NT, self.dt)
            A = Agent_3D(i, ref_pts, self.v_max, self.vz_max, self.w_max, self.controlFrequency)
            self.agents.append(A)

    def control(self, start_point, trajectories):  # The start_point and trajectories here refer to all robots
        self.agents = []
        self.iniPosition_ininitialmap = np.array(start_point).reshape((-1, 3))
        trajectories = np.array(trajectories).reshape((self.num_robots, len(trajectories[0]) * 3))
        time_interval_mat = np.zeros((self.Ncar, int(trajectories.shape[1] / 3 - 1)))

        for i in range(self.Ncar):
            for j in range(0, trajectories.shape[1] - 5, 3):
                time_interval_mat[i, j // 3] = np.ceil(
                    np.linalg.norm(trajectories[i, j:j + 3] -
                                trajectories[i, j + 3:j + 6]) / self.v_max * 4 / 3)

        for i in range(self.Ncar):
            traj_i   = trajectories[i].reshape(-1, 3)               # 不再手动 vstack
            ref_pts  = referencePoints(traj_i,
                                    time_interval_mat[i],            # 对应机器人自己的时间数组
                                    self.NT, self.dt)
            A = Agent_3D(i, ref_pts, self.v_max, self.vz_max, self.w_max, self.controlFrequency)
            self.agents.append(deepcopy(A))

        '''
        self.constraintSeries = []
        for i, agent in enumerate(self.agents):
            constraints = []
            rp = agent.rp
            for j in range(rp.shape[0]):
                if not constraints or np.any(np.dot(self.polySeries[constraints[-1]]['A'], rp[j]) >= self.polySeries[constraints[-1]]['b']):
                    for k in range(len(self.polySeries)):
                        if np.all(np.dot(self.polySeries[k]['A'], rp[j]) < self.polySeries[k]['b']):
                            constraints.append(k)
                            break
            self.constraintSeries.append(constraints)
        '''

        self.A = np.array([
            [1, self.dt, 0.5 * self.dt ** 2, 0, 0, 0, 0, 0, 0],
            [0, 1, self.dt, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, self.dt, 0.5 * self.dt ** 2, 0, 0, 0],
            [0, 0, 0, 0, 1, self.dt, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, self.dt, 0.5 *self.dt **2],
            [0, 0, 0, 0, 0, 0, 0, 1, self.dt],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])                                             #状态转移矩阵x(t+1)=Ax(t) 状态量x=[x,vx,ax,y,vy,ay,z,vz,az]T

        self.B = np.array([
            [1 / 6 * self.dt ** 3, 0, 0],
            [0.5 * self.dt ** 2, 0, 0],
            [self.dt, 0, 0],
            [0, 1 / 6 * self.dt ** 3, 0],
            [0, 0.5 * self.dt ** 2, 0],
            [0, self.dt, 0],                            #控制输入矩阵Bu（t）      控制量u=[jx,jy,jz] 
            [0, 0, 1/6 * self.dt ** 3],
            [0, 0, 0.5 * self.dt ** 2],
            [0, 0, self.dt]
        ])                                             
         #状态权重矩阵
        Q = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0.5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0.1, 0, 0, 0, 0, 0, 0],       
            [0, 0, 0, 1, 0, 0, 0, 0, 0],                
            [0, 0, 0, 0, 0.5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0.5, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0.1]
        ])

        R = np.array([
            [0.1, 0, 0],                                #控制权重矩阵
            [0, 0.1, 0],
            [0, 0, 0.1]
        ])

        # 状态约束矩阵，确定各状态量上下界
        self.Fx0 = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1]
        ])
        # 状态转移矩阵的累积效应
        self.AX = np.zeros(((self.N + 1) * self.n, self.n))
        for i in range(self.N + 1):
            self.AX[i * self.n:(i + 1) * self.n, :] = np.linalg.matrix_power(self.A, i)

        # 控制输入矩阵的累积效应
        self.BU = np.zeros(((self.N + 1) * self.n, self.N * self.m))
        for i in range(self.N + 1):
            for j in range(self.N):
                if i > j:        #控制输入uj经过i-j-1步状态转移累积效应
                    self.BU[i * self.n:(i + 1) * self.n, j * self.m:(j + 1) * self.m] = np.linalg.matrix_power(self.A,
                                                                                                               i - j - 1) @ self.B
                else:
                    self.BU[i * self.n:(i + 1) * self.n, j * self.m:(j + 1) * self.m] = np.zeros((self.n, self.m))


        self.x_of_agent = [None] * len(self.agents)                                                  # self.x_of_agent[i]表示第i个智能体的状态矩阵
        self.referencelist = [[None]*(self.NT + self.N) for _ in range(len(self.agents))]
        self.gx = [[None]*(self.NT + self.N) for _ in range(len(self.agents))]
        for i, agent in enumerate(self.agents):
            for k in range(self.NT + 1):
                for t in range(self.N):
                    if t + k + 1 >= agent.rp.shape[0]:
                        # 尾部填一个极大约束，或把上一帧值沿用
                        self.gx[i][t + k] = np.full(18, np.inf)
                        self.referencelist[i][t + k] = np.zeros(self.n)
                        continue

                    # ---------- 参考速度 ----------
                    rp_diff = np.linalg.norm(agent.rp[t + k + 1] - agent.rp[t + k])
                    rp_diff = max(rp_diff, 1e-6)                         # 防 0
                    vr = (agent.rp[t + k + 1] - agent.rp[t + k]) / rp_diff * min(rp_diff / self.dt, agent.vmax*0.75)

                    # ---------- 参考状态 ----------
                    xr = np.array([agent.rp[t + k, 0], vr[0], 0,
                                agent.rp[t + k, 1], vr[1], 0,
                                agent.rp[t + k, 2], vr[2], 0])
                    self.referencelist[i][t + k] = xr

                    # ---------- 状态约束 ----------
                    constraint_vec = np.array([
                        self.xb,  agent.vmax, self.a_max,
                    -self.xa,  agent.vmax, self.a_max,
                        self.yb,  agent.vmax, self.a_max,
                    -self.ya,  agent.vmax, self.a_max,
                        self.zb,  agent.vmax, self.a_max,
                    -self.za,  agent.vmax, self.a_max
                    ])
                    self.gx[i][t + k] = constraint_vec - self.Fx0 @ xr

            x = np.zeros((self.n, self.NT + 1))
            x[:, 0] = np.array(
                [self.agents[i].initialPosition[0], self.agents[i].velocity[0], 0, self.agents[i].initialPosition[1], self.agents[i].velocity[1], 0, self.agents[i].initialPosition[2],self.agents[i].velocity[2], 0])  #初始化状态
            self.x_of_agent[i] = x
            self.agents[i].futureState = np.dot(self.AX, x[:, 0])


        #print('agent state initialization finished')
        #控制约束矩阵
        Fu = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1]
        ])   
        gu = np.array([self.jerk, self.jerk, self.jerk, self.jerk, self.jerk, self.jerk])  #  Fu@u <=gu

        self.u = np.zeros((self.m, self.NT))
        self.QX = np.copy(Q)      #状态权重
        self.RU = np.copy(R)      #控制权重
        self.FU = np.copy(Fu)     #状态约束
        self.gU = np.copy(gu)     #控制约束

        # QX, RU, FU, gU
        for i in range(self.N - 1):
            self.QX = block_diag(self.QX, Q)
            self.RU = block_diag(self.RU, R)
            self.FU = block_diag(self.FU, Fu)
            self.gU = np.hstack((self.gU, gu))
        self.QX = block_diag(self.QX, Q)
        self.H = block_diag(self.QX, self.RU)

        self.time_list = np.zeros((self.NT, 3))
        self.exit_flag_data = np.zeros((self.Ncar, self.NT))
        self.nextposelist = [[None] * self.NT for _ in range(self.Ncar)]
        self.nextposelist_controller = [[] * self.NT for _ in range(self.Ncar)]
        self.omegalist = np.zeros((self.Ncar, self.NT))
        self.solutiondata = [[None] * self.NT for _ in range(self.Ncar)]
        self.solutioncondition = [[None] * self.NT for _ in range(self.Ncar)]
        self.sendtime = 0
        self.lastslacknum = 0

        self.timelist_qp = np.zeros((self.Ncar, self.NT))
        self.timelist_sensorrange = np.zeros((self.Ncar, self.NT))
        self.timelist_all = np.zeros((self.Ncar, self.NT))
        self.timelist_update = np.zeros((self.Ncar, self.NT))
        self.timelist_read = np.zeros((self.Ncar, self.NT))
        self.timelist_multithread = np.zeros(self.NT)
        self.timelist_polytope = np.zeros((self.Ncar, self.NT))
        # lock = Lock()
        v1_list = np.zeros(self.Ncar)
        v_real_list = np.zeros(self.Ncar)
        # lastz = None
        lastslacknum = 0

        actualState = np.zeros((self.Ncar, 12))
        for i in range(self.Ncar):
            actualState[i][0] = self.iniPosition_ininitialmap[i][0]
            actualState[i][3] = self.iniPosition_ininitialmap[i][1]
            actualState[i][6] = self.iniPosition_ininitialmap[i][2]
        self.z_of_agents = [[] for _ in range(self.num_robots)]

# 定义智能体线程函数，处理单个智能体的MPC控制逻辑
def agent_thread_3D(agent, Controller, agent_index, actualState, k, lastz, lastslacknum):
    """
    单个智能体的MPC控制线程：
    - 构造 QP：min 1/2 z^T H z  s.t.
        A_eq z = b_eq       （动力学等式：X = AX x0 + BU U）
        F z <= g            （状态/控制不等式，含可选松弛）
    - 这里 z = [X; U; (slack?)]
    """
    for j in range(Controller.NT + 1):
        for t in range(Controller.N):
            if t + j + 1 >= agent.rp.shape[0]:
                # 尾部填一个极大约束，或把上一帧值沿用
                Controller.gx[agent_index][t + j] = np.full(18, np.inf)
                Controller.referencelist[agent_index][t + j] = np.zeros(Controller.n)
                continue

            # ---------- 参考速度 ----------
            rp_diff = np.linalg.norm(agent.rp[t + j + 1] - agent.rp[t + j])
            rp_diff = max(rp_diff, 1e-6)                         # 防 0
            vr = (agent.rp[t + j + 1] - agent.rp[t + j]) / rp_diff * min(rp_diff / Controller.dt, agent.vmax*0.75)

            # ---------- 参考状态 ----------
            xr = np.array([agent.rp[t + j, 0], vr[0], 0,
                        agent.rp[t + j, 1], vr[1], 0,
                        agent.rp[t + j, 2], vr[2], 0])
            Controller.referencelist[agent_index][t + j] = xr

            # ---------- 状态约束 ----------
            constraint_vec = np.array([
                Controller.xb,  agent.vmax, Controller.a_max,
            -Controller.xa,  agent.vmax, Controller.a_max,
                Controller.yb,  agent.vmax, Controller.a_max,
            -Controller.ya,  agent.vmax, Controller.a_max,
                Controller.zb,  agent.vmax, Controller.a_max,
            -Controller.za,  agent.vmax, Controller.a_max
            ])
            Controller.gx[agent_index][t + j] = constraint_vec - Controller.Fx0 @ xr
    all_time = time.time()

    # --- 读取当前全局状态到 agent 对象 ---
    # [x,vx,ax, y,vy,ay, z,vz,az, yaw,roll,pitch]
    agent.velocity     = [actualState[agent_index][1], actualState[agent_index][4], actualState[agent_index][7]]
    agent.position     = [actualState[agent_index][0], actualState[agent_index][3], actualState[agent_index][6]]
    agent.acceleration = [actualState[agent_index][2], actualState[agent_index][5], actualState[agent_index][8]]
    agent.theta        = [actualState[agent_index][9], actualState[agent_index][10], actualState[agent_index][11]]

    # --- 首次使用 path 时兜底初始化 ---
    if not hasattr(agent, "path") or agent.path is None or len(np.shape(agent.path)) != 2:
        agent.path = np.array(agent.position, dtype=float).reshape(1, 3)
    agent.path = np.vstack((agent.path, np.array(agent.position, dtype=float).reshape(1, 3)))

    time_node1 = time.time()
    Controller.timelist_read[agent_index, k] = time_node1 - all_time

    # 记录角度与状态
    agent.theta = np.append(agent.theta, [actualState[agent_index][9], actualState[agent_index][10], actualState[agent_index][11]])
    Controller.x_of_agent[agent_index][:, k] = actualState[agent_index][:Controller.n]

    xk = np.copy(Controller.x_of_agent[agent_index][:, k])  # 当前时刻状态（作为等式右端的自由响应项）

    # ========== 参考轨迹处理 ==========
    # 用于构造参考状态 xr（可用于代价/不等式约束；动力学等式不直接用它）
    rp_diff = float(np.linalg.norm(agent.rp[k + 1] - agent.rp[k]))
    if rp_diff < 1e-9:
        rp_diff = 1e-9  # 防 0
    vr = (agent.rp[k + 1] - agent.rp[k]) / Controller.dt
    vmax_ref = agent.vmax * 0.75
    if np.linalg.norm(vr) > vmax_ref:
        vr = (agent.rp[k + 1] - agent.rp[k]) / rp_diff * vmax_ref

    xr = np.array([
        agent.rp[k, 0], vr[0], 0,
        agent.rp[k, 1], vr[1], 0,
        agent.rp[k, 2], vr[2], 0
    ], dtype=float)  # 参考状态；不直接进入等式约束

    # =========== 约束初始化（单步模板） ============
    FX = Controller.Fx0                      # (18 x 9) 单步状态约束模板
    Fu_big = Controller.FU                   # (6N x 3N) 已经按时域展开好的控制约束
    gU_big = Controller.gU                   # (6N,)

    # =========== 将状态约束沿时域展开到 (N+1) 步 ============
    # FX_block: ((N+1)*18) x ((N+1)*9) = 18*(N+1) x 63
    FX_blocks = [FX] * (Controller.N + 1)
    FX_block = block_diag(*FX_blocks)

    # gX_block: 把 control() 里准备好的每一步 gX 堆起来
    # Controller.gx[agent_index][t+k] 是长度 18 的向量
    gX_list = []
    for t in range(Controller.N + 1):
        # 防止越界：若轨迹尾部不足，control() 里已填充了 inf，这里直接取
        gX_list.append(Controller.gx[agent_index][t + k])
    gX_block = np.hstack(gX_list)           # 长度 18*(N+1)

    # =========== 拼成对齐到 z=[X;U] 的不等式 ============
    # 状态部分： [ FX_block  |  0_( (N+1)*18 × N*m ) ]
    F_state = np.hstack([
        FX_block,
        np.zeros((FX_block.shape[0], Controller.N * Controller.m))
    ])
    g_state = gX_block

    # 控制部分： [ 0_(6N × (N+1)*n)  |  Fu_big ]
    F_ctrl = np.hstack([
        np.zeros((Fu_big.shape[0], (Controller.N + 1) * Controller.n)),
        Fu_big
    ])
    g_ctrl = gU_big

    # 总不等式：纵向拼接
    Fnew = np.vstack([F_state, F_ctrl])      # 行数 = 18*(N+1) + 6N = 18*7 + 36 = 162
    gnew = np.hstack([g_state, g_ctrl])      # 长度同上

    # ===========（可选）松弛变量：在列右侧补 slacknum 列零 ============
    slacknum = 0
    Hnew = block_diag(Controller.H, np.eye(slacknum) * Controller.penalty)

    if slacknum > 0:
        Fnew = np.hstack([Fnew, np.zeros((Fnew.shape[0], slacknum))])
        # 若需要对某些行添加 +slack_i，请在此处逐行置 1（略）
        # 追加 slack >= 0
        F_slack_pos = np.hstack([np.zeros((slacknum, Hnew.shape[0])), np.eye(slacknum)])
        Fnew = np.vstack([Fnew, F_slack_pos])
        gnew = np.hstack([gnew, np.zeros(slacknum)])

    # =========== 等式约束（整段动力学） ============
    xr_stack = xr
    for ii in range(Controller.N):
        xr_stack = np.hstack([xr_stack, xr])
    b_eq = Controller.AX @ xk - xr_stack
    A_eq = np.hstack([
        np.eye((Controller.N + 1) * Controller.n),
        -Controller.BU
    ])
    if slacknum > 0:
        A_eq = np.hstack([A_eq, np.zeros(((Controller.N + 1) * Controller.n, slacknum))])

    # =========== 建立并求解 QP ============
    problem = Problem(
        Hnew,
        np.zeros(Hnew.shape[0]),
        Fnew, gnew,
        A_eq, b_eq
    )
    qp_per_start_time = time.time()
    # 优先 mosek，失败时依次尝试 osqp、quadprog、scs
    solution = None
    for solver in ("mosek", "osqp", "quadprog", "scs"):
        try:
            sol = solve_problem(problem, solver=solver)
            if sol is not None and sol.x is not None:
                solution = sol
                break
        except Exception:
            continue
    z = solution.x if (solution is not None) else None
    Controller.timelist_qp[agent_index, k] = time.time() - qp_per_start_time

    # 失败回退
    if z is None or (hasattr(z, '__len__') and len(z) == 0):
        if Controller.lastz is not None and (hasattr(Controller.lastz, '__len__') and len(Controller.lastz) > 0):
            z = Controller.lastz
            slacknum = Controller.lastslacknum
        elif lastz is not None and (hasattr(lastz, '__len__') and len(lastz) > 0):
            z, slacknum = lastz, lastslacknum
            Controller.lastz = z
            Controller.lastslacknum = slacknum
        else:
            # 无可用解：使用零控制
            z = None
        if z is None:
            print(f'agent {agent_index} QP fail (no fallback)! pos={agent.position}, using zero control')
            Controller.u[:, k] = np.zeros(Controller.m)
            agent.velocity = [0.0, 0.0, 0.0]
            agent.acceleration = [actualState[agent_index][2], actualState[agent_index][5], actualState[agent_index][8]]
            agent.position = [actualState[agent_index][0], actualState[agent_index][3], actualState[agent_index][6]]
            return actualState[agent_index], Controller.v[agent_index], Controller.w[agent_index], agent, Controller.lastz, Controller.lastslacknum
    else:
        Controller.lastz = z
        Controller.lastslacknum = slacknum

    # 当前时刻控制输入（首个 u_k）
    u_start = (Controller.N + 1) * Controller.n
    Controller.u[:, k] = z[u_start:u_start + Controller.m]

    # === 写回/推进状态 ===
    Controller.z_of_agents[agent_index] = (
        z[(Controller.N + 1) * Controller.n:] if slacknum == 0
        else z[(Controller.N + 1) * Controller.n:(-1) * slacknum]
    )

    # 预测状态推进一拍：x_{k+1} = A x_k + B u_k
    Controller.x_of_agent[agent_index][:, k + 1] = (
        Controller.A @ xk + Controller.B @ Controller.u[:, k]
    )
    currentState = Controller.x_of_agent[agent_index][:, k + 1]

    # 物理控制器输出（根据你现有 controller(...) 接口）
    actualState[agent_index], Controller.v[agent_index], Controller.w[agent_index] = controller(
        agent, Controller.z_of_agents[agent_index][0:Controller.m], Controller.n
    )

    # 未来预测状态（用于可视化/日志）
    agent.futureState = (
        Controller.AX[:Controller.N * Controller.n, :] @ currentState
        + Controller.BU[:Controller.N * Controller.n, :Controller.N * Controller.m - Controller.m]
        @ Controller.z_of_agents[agent_index][Controller.m:]
    )

    # 回写 agent 的物理量（便于外部读取）
    agent.acceleration = [actualState[agent_index][2], actualState[agent_index][5], actualState[agent_index][8]]
    agent.velocity     = [actualState[agent_index][1], actualState[agent_index][4], actualState[agent_index][7]]
    agent.position     = [actualState[agent_index][0], actualState[agent_index][3], actualState[agent_index][6]]

    return actualState[agent_index], Controller.v[agent_index], Controller.w[agent_index], agent, Controller.lastz, Controller.lastslacknum
def referencePoints(points, time, NT, dt):
    """
    根据关键路径点生成离散参考轨迹（只支持 3D）。
    ---------------------------------------------------------------
    points : (M, 3) ndarray，关键点 [x, y, z]
    time   : (M,)  ndarray，各段耗时
    NT     : 目标步数
    dt     : 控制周期
    返回值 : (NT + 100, 3) ndarray
    ---------------------------------------------------------------
    """
    points = np.asarray(points, dtype=float)
    time   = np.asarray(time,   dtype=float)

    # —— 1. 必须是 3 维 ——
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"`points` 必须是 (M, 3) 的三维坐标数组，收到 shape={points.shape}")

    ref_segments = []                     # 存每一段的插值结果
    for i in range(points.shape[0] - 1):
        step = max(1, int(round(time[i] / dt)))           # 至少插 1 步
        segment = np.linspace(points[i], points[i + 1],
                              step, endpoint=False)       # 避免首尾重复
        ref_segments.append(segment)

    referencePoints = np.vstack(ref_segments)             # shape (?, 3)

    # —— 2. 末尾停在终点，补足步数 ——#
    remaining = int(max(0, NT - referencePoints.shape[0] + 100))
    if remaining > 0:
        tail = np.tile(points[-1], (remaining, 1))        # (remaining, 3)
        referencePoints = np.vstack((referencePoints, tail))

    return referencePoints

def remove_after_close_points(array, threshold=0.01):
    for i in range(len(array) - 1):
        norm_diff = np.linalg.norm(array[i] - array[i + 1])
        if norm_diff < threshold:
            return array[:i + 1]
    return array
