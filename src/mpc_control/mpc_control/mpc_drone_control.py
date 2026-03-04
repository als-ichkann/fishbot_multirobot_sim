#!/usr/bin/env python3
"""
MPC Drone Control ROS2 Node

订阅 gt/odom、trajectory (apf_trajectory)，执行 MPC 轨迹跟踪，发布 cmd_vel (Twist)。
用于控制仿真中的无人机，运动接口符合 geometry_msgs/Twist。
"""

import numpy as np
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


class MPCDroneControlNode(Node):
    """单机器人 MPC 轨迹跟踪节点"""

    def __init__(self):
        super().__init__("mpc_drone_control")
        self.declare_parameter("control_dt", 0.1)
        self.declare_parameter("control_frequency", 10)
        self.declare_parameter("velocity_scale", 500.0)
        self.declare_parameter("min_speed", 0.15)
        dt = float(self.get_parameter("control_dt").value)
        self._dt = dt
        self._control_frequency = float(self.get_parameter("control_frequency").value)
        self._velocity_scale = float(self.get_parameter("velocity_scale").value)
        self._min_speed = float(self.get_parameter("min_speed").value)

        # 共享状态
        self._lock = threading.Lock()
        self._state = np.zeros(12)  # [x,vx,ax, y,vy,ay, z,vz,az, yaw,roll,pitch]
        self._state_valid = False
        self._trajectory_points = None  # (N,3) ndarray
        self._new_trajectory_flag = False
        self._mpc = None
        self._mpc_initialized = False
        self._current_step = 0
        self._lastz = None
        self._lastslacknum = 0
        self._actual_state = np.zeros((1, 12))
        self._num_robots = 1

        # 订阅 / 发布
        self._odom_sub = self.create_subscription(
            Odometry, "gt/odom", self._odom_cb, 10
        )
        self._traj_sub = self.create_subscription(
            Path, "trajectory", self._trajectory_cb, 10
        )
        self._cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)

        # 控制定时器 (10 Hz)
        self._timer = self.create_timer(1.0 / self._control_frequency, self._control_timer_cb)

        self.get_logger().info("mpc_drone_control node started")

    def _quat_to_yaw(self, q):
        """从四元数 (x,y,z,w) 提取 yaw（绕 z 轴）"""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _odom_cb(self, msg):
        """从 Odometry 更新 state"""
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        yaw = self._quat_to_yaw(q)
        with self._lock:
            self._state[0] = p.x
            self._state[3] = p.y
            self._state[6] = p.z
            self._state[1] = vx
            self._state[4] = vy
            self._state[7] = vz
            self._state[9] = yaw
            self._state_valid = True

    def _trajectory_cb(self, msg):
        """收到新 Path 时更新参考轨迹，并标记可中断当前执行"""
        if len(msg.poses) < 2:
            self.get_logger().warn("Trajectory has fewer than 2 poses, ignored")
            return
        pts = np.array([
            [p.pose.position.x, p.pose.position.y, p.pose.position.z]
            for p in msg.poses
        ])
        with self._lock:
            self._trajectory_points = pts
            self._new_trajectory_flag = True

    def _build_mpc_from_trajectory(self, current_pos, trajectory_raw):
        """根据当前位置和原始轨迹，构建/更新 MPC"""
        from . import robot_3D

        clipped = robot_3D.get_3d_trajectories(current_pos, [trajectory_raw])
        if len(clipped[0]) == 0:
            self.get_logger().warn("Trajectory clip result empty")
            return False

        traj_clipped = clipped[0]
        pos = np.array(current_pos).reshape(-1, 3)
        # 构造 start_trajectory：当前位置 -> 轨迹首点
        start_traj = np.vstack((pos, traj_clipped[0:1]))
        v_max = 0.3
        NT = 1000
        dt = self._dt

        n_seg = max(1, start_traj.shape[0] - 1)
        time_interval = np.zeros(n_seg)
        for j in range(n_seg):
            diff = start_traj[j + 1] - start_traj[j]
            time_interval[j] = np.ceil(np.linalg.norm(diff) / v_max * 4 / 3)
        time_interval = np.maximum(time_interval, 1.0)
        ref_pts = robot_3D.referencePoints(start_traj, time_interval, NT, dt)
        ref_pts = robot_3D.remove_after_close_points(ref_pts, 1e-4)

        # 拼接：平滑段 + 截取轨迹
        rp = np.vstack((ref_pts[:-1], traj_clipped))
        if len(rp) < 2:
            rp = np.vstack((rp, rp[-1:]))

        # MPC.control() 期望: start_point=[[x,y,z]], trajectories=[[[x,y,z],[x,y,z],...]]
        start_point = [list(current_pos)]
        trajectories = [rp.tolist()]

        if self._mpc is None:
            # 首次创建：需要 dummy discrete_points 满足 __init__，每行至少 2 个点 (6 个标量) 以计算 time_interval
            dummy = [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]]]
            self._mpc = robot_3D.MPC_3D(
                num_robots=1, N=6, dt=dt, discrete_points=dummy
            )

        self._mpc.control(start_point, trajectories)
        self._current_step = 0
        self._actual_state[0] = np.copy(self._state)
        self._mpc.actualState = np.zeros((1, 12))
        self._mpc.actualState[0] = np.copy(self._state)
        # 同步 agent 初始状态
        agent = self._mpc.agents[0]
        agent.position = np.array(current_pos).reshape(-1, 3)
        agent.velocity = [self._state[1], self._state[4], self._state[7]]
        agent.acceleration = [self._state[2], self._state[5], self._state[8]]
        agent.theta = [
            self._state[9], self._state[10], self._state[11]
        ]
        self._mpc_initialized = True
        return True

    def _control_timer_cb(self):
        """控制定时器：执行单步 MPC，发布 cmd_vel"""
        with self._lock:
            if not self._state_valid:
                return
            state = np.copy(self._state)
            trajectory_points = self._trajectory_points
            new_flag = self._new_trajectory_flag
            self._new_trajectory_flag = False

        if trajectory_points is None or len(trajectory_points) < 2:
            # 无轨迹：发布零速
            twist = Twist()
            self._cmd_pub.publish(twist)
            return

        current_pos = [state[0], state[3], state[6]]

        with self._lock:
            if new_flag or not self._mpc_initialized:
                ok = self._build_mpc_from_trajectory(current_pos, trajectory_points)
                if not ok:
                    twist = Twist()
                    self._cmd_pub.publish(twist)
                    return
                # 新轨迹已应用，直接执行第一步
            mpc = self._mpc
            if mpc is None or not self._mpc_initialized:
                twist = Twist()
                self._cmd_pub.publish(twist)
                return

        agent = mpc.agents[0]
        NT = mpc.NT

        if self._current_step >= NT:
            # 轨迹已完成
            v = np.array(getattr(agent, "velocity", [0, 0, 0]), dtype=float) * self._velocity_scale
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-9 and v_norm < self._min_speed:
                v = v / v_norm * self._min_speed
            if v_norm > 0.5:
                v = v / v_norm * 0.5
            ov = np.nan_to_num(np.array(getattr(agent, "angular_velocity", [0, 0, 0])), nan=0.0, posinf=0.0, neginf=0.0)
            twist = Twist()
            twist.linear.x = float(v[0])
            twist.linear.y = float(v[1])
            twist.linear.z = float(v[2])
            twist.angular.x = float(ov[0])
            twist.angular.y = float(ov[1])
            twist.angular.z = float(ov[2])
            self._cmd_pub.publish(twist)
            return

        # 更新其他 agent 位置（单机时仅自身）
        state_list = {"state_0": state}
        for i in range(mpc.num_robots):
            mpc.agents[i].position = np.array([state_list[f"state_{i}"][0], state_list[f"state_{i}"][3], state_list[f"state_{i}"][6]]).reshape(-1, 3)

        actual_state = np.zeros((mpc.num_robots, 12))
        actual_state[0] = state
        k = self._current_step

        try:
            from . import robot_3D

            _, _, _, agent, self._lastz, self._lastslacknum = robot_3D.agent_thread_3D(
                agent, mpc, 0, actual_state, k, self._lastz, self._lastslacknum
            )
        except Exception as e:
            self.get_logger().error(f"agent_thread_3D failed: {e}")
            twist = Twist()
            self._cmd_pub.publish(twist)
            return

        # 更新本地状态
        with self._lock:
            self._state = np.copy(actual_state[0])
            self._current_step += 1

        # 发布 Twist：放大 MPC 输出并施加最小速度（MPC 输出过小导致无人机不动）
        v = np.array(agent.velocity, dtype=float)
        if not np.isfinite(v).all():
            self.get_logger().warn_throttle(1.0, "MPC velocity contains NaN/inf, clamping to 0")
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        v = v * self._velocity_scale
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-9 and v_norm < self._min_speed:
            v = v / v_norm * self._min_speed
        v_norm = np.linalg.norm(v)
        if v_norm > 0.5:  # 限制最大速度
            v = v / v_norm * 0.5
        ov = np.array(getattr(agent, "angular_velocity", [0.0, 0.0, 0.0]), dtype=float)
        if not np.isfinite(ov).all():
            self.get_logger().warn_throttle(1.0, "MPC angular_velocity contains NaN/inf, clamping to 0")
        ov = np.nan_to_num(ov, nan=0.0, posinf=0.0, neginf=0.0)
        twist = Twist()
        twist.linear.x = float(v[0])
        twist.linear.y = float(v[1])
        twist.linear.z = float(v[2])
        twist.angular.x = float(ov[0])
        twist.angular.y = float(ov[1])
        twist.angular.z = float(ov[2])
        self._cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = MPCDroneControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"mpc_drone_control exited: {e}")
        import traceback
        traceback.print_exc()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
