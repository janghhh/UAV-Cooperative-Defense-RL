# Note: This code is based on a previously published study and is used as a baseline for comparison with the proposed method.

import gymnasium as gym
import numpy as np
import airsim
import math
import time
import random
from collections import deque


class AirSimMultiDroneEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2"),
        port=41451,
        step_length=0.01,
        leader_velocity=3.0,
        optimal_distance=10.0,
        far_cutoff=60.0,
        too_close=0.5,
        dt=0.15,
        do_visualize=True
    ):
        super().__init__()
        self.possible_agents = list(follower_names)
        self.agents = self.possible_agents[:]

        # 충돌 관련 설정
        self.COLLISION_THRESHOLD = 1.5
        self.eliminated_agents = set()
        self.episode_ally_collision = False
        self.STOP_DISTANCE_LEADER_OBSTACLE = 1.0

        # 속도/액션 버퍼
        self.vmax_self = 2.0
        self._timestep = float(dt)

        # 에이전트/리더 속도 산출용 버퍼
        self._last_pose = {}
        self._last_time = {}

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self.success_steps = {a: 0 for a in self.possible_agents}

        # 하이퍼파라미터/환경 파라미터
        self.step_length = float(step_length)
        self.fixed_z = -10.0
        self.dt = float(dt)
        self.do_visualize = bool(do_visualize)
        self.max_cmd_speed = self.step_length / self.dt
        self.leader_velocity = float(leader_velocity)
        self.optimal_distance = float(optimal_distance)
        self.far_cutoff = float(far_cutoff)
        self.too_close = float(too_close)
        self.follower_names = list(follower_names)
        # Lidar 메모리: { "AgentName": { "TargetName": [dx, dy] } }
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        # [추가] Lidar 센서 이름 매핑 (settings.json의 이름과 일치해야 함)
        self.lidar_names = {
            agent: f"{agent}_LidarSensor" for agent in self.possible_agents
        }

        # States
        self.step_count = 0
        self.episode_count = 0

        # 통계 계산을 위한 Deque
        self.stats_history = {
            "win": deque(maxlen=20),
            "coll_leader": deque(maxlen=20),
            "coll_drone": deque(maxlen=20),
            "coll_obj": deque(maxlen=20)
        }

        # 동적 장애물 관련
        self.isIdle = None
        self.D_O_STATE = {0: "idle", 1: "attack"}

        # ===== obs / act / share_obs spaces =====
        self.K_ally = len(follower_names) - 1
        self.K_enemy = 1
        self.num_ally = self.K_ally
        self.num_enemy = self.K_enemy

        # ---- [수정] 관측 공간 범위 정의 (상대 좌표 도입) ----
        # 기존: Bearing(-1~1), Dist(0~200)
        # 변경: Rel_X(-1~1), Rel_Y(-1~1) 로 변경 (100m 기준 정규화)
        
        low_rel_pos = -1.0
        high_rel_pos = 1.0
        
        low_vel = -1.0
        high_vel = 1.0
        low_rate = -1.0
        high_rate = 1.0
        low_self_state = -1.0
        high_self_state = 1.0

        # [리더(2)] + [아군(2)*K] + [적(4)*K] + [self_state(2)]
        # 적(Enemy)의 경우: [rel_x, rel_y, closing_speed, los_rate]
        per_agent_low = (
            [low_rel_pos, low_rel_pos] +                          # Leader (dx, dy)
            [low_rel_pos, low_rel_pos] * self.num_ally +          # Allies (dx, dy)
            [low_rel_pos, low_rel_pos, low_vel, low_rate] * self.num_enemy + # Enemy
            [low_self_state] * 2                                  # Self (vx, vy)
        )
        per_agent_high = (
            [high_rel_pos, high_rel_pos] +
            [high_rel_pos, high_rel_pos] * self.num_ally +
            [high_rel_pos, high_rel_pos, high_vel, high_rate] * self.num_enemy +
            [high_self_state] * 2
        )

        obs_dim = len(per_agent_low)
        share_obs_dim = obs_dim * len(self.possible_agents)

        self.observation_spaces = {
            agent: gym.spaces.Box(
                low=np.array(per_agent_low, dtype=np.float32),
                high=np.array(per_agent_high, dtype=np.float32),
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.MAX_YAW = math.radians(90)
        self.MAX_SPEED = 10

        self.action_spaces = {
            agent: gym.spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(2,),
                dtype=np.float64,
            )
            for agent in self.possible_agents
        }

        self.share_observation_spaces = gym.spaces.Box(
            low=np.array(per_agent_low * len(self.possible_agents), dtype=np.float32),
            high=np.array(per_agent_high * len(self.possible_agents), dtype=np.float32),
            shape=(share_obs_dim,),
            dtype=np.float32,
        )

        self.dynamic_name = "DynamicObstacle0"

        # PN 보상용 버퍼들
        self._prev_d_leader_enemy = None
        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._prev_los_angle = {a: None for a in self.possible_agents}

        # PN-style reward parameters
        self.REWARD_HIT_ENEMY = 100.0   # 성공 보상은 크게
        self.REWARD_LEADER_HIT = -250.0
        self.REWARD_AGENT_CRASH = -500.0 # [중요] 자살 방지 위해 충돌 패널티 대폭 강화

        self.W_CLOSE = 0.5
        self.W_LOS = 0.5
        self.W_DIST = 0.5
        self.W_ALLY = 0.5
        self.STEP_PENALTY = 0.0 # [중요] 시간 패널티 제거 (오래 살아남아 요격하도록)

        self.MAX_DIST_DELTA = 5.0
        self.MAX_ANGLE_DELTA = math.radians(20.0)
        self.ALLY_SAFE_DIST = 5.0

        self.REAL_LEADER_SPEED = 1.5
        self.leader_move_timer = 0
        self.current_leader_vel = [0.0, 0.0]

        # 위치 캐시
        self.start_location = {}
        self.current_location = {}

        self.obs_speed_min = 2.0
        self.obs_speed_max = 12.0
        self.dynamic_obstacle_speed = 6.0  # 초기값

        # --- [추가] 평가 지표(IR, SIR) 계산용 변수 ---
        self.MISS_THRESHOLD = 5.0  # 5m 이내로 접근했다가 멀어지면 '1차 요격 실패(Miss)'로 간주
        self.primary_miss_occurred = False
        self._prev_dist_to_enemy = {a: 999.0 for a in self.possible_agents}
        # 클라이언트 셋업
        self.client = airsim.MultirotorClient(ip=ip_address, port=port)
        self.client.confirmConnection()

        self._last_visualize_t = time.time()

    def _get_lidar_measurement(self, agent_name, target_name):
        """
        Lidar로 타겟을 관측합니다.
        - 감지 성공: 포인트 클라우드 평균 위치 반환 + 메모리 갱신
        - 감지 실패: 메모리에 저장된 '마지막 위치' 반환
        """
        if agent_name in self.eliminated_agents or target_name in self.eliminated_agents:
            return 0.0, 0.0, False
        lidar_name = self.lidar_names[agent_name]
        lidar_data = self.client.getLidarData(lidar_name, vehicle_name=agent_name)
        
        # 1. 포인트 데이터 파싱
        # SensorLocalFrame 사용 시: points[:, 0]=전방(x), points[:, 1]=우측(y)
        points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
        
        detected = False
        meas_dx, meas_dy = 0.0, 0.0

        if len(points) > 2:
            # 2. 객체 인식 시뮬레이션 (GT 위치 근처의 포인트만 필터링)
            # 현재 Yaw가 0으로 고정된 시뮬레이션이므로, World Frame 차이 ≈ Body Frame 차이로 근사 가능
            # (만약 드론이 회전한다면 회전 행렬 적용 필요)
            
            my_pos = self.current_location[agent_name].position
            tgt_pos = self.current_location[target_name].position
            
            gt_dx = tgt_pos.x_val - my_pos.x_val
            gt_dy = tgt_pos.y_val - my_pos.y_val
            
            # Lidar 포인트들 중 실제 타겟 위치 반경 3m 내에 있는 점만 추출
            # (SensorLocalFrame이므로 points 자체가 상대좌표임)
            dist_sq = (points[:, 0] - gt_dx)**2 + (points[:, 1] - gt_dy)**2
            mask = dist_sq < (3.0)**2  
            target_points = points[mask]

            if len(target_points) > 0:
                # 감지 성공! 포인트들의 평균을 측정값으로 사용
                mean_pos = np.mean(target_points, axis=0)
                raw_dx = mean_pos[0]
                raw_dy = mean_pos[1]
                
                # 정규화 (-1 ~ 1, 100m 기준)
                norm_dx = np.clip(raw_dx / 100.0, -1.0, 1.0)
                norm_dy = np.clip(raw_dy / 100.0, -1.0, 1.0)
                
                # 메모리 갱신
                self.lidar_memory[agent_name][target_name] = [float(norm_dx), float(norm_dy)]
                detected = True
        
        # 3. 반환 (감지했으면 갱신된 값, 못했으면 메모리 값)
        final_val = self.lidar_memory[agent_name][target_name]
        
        # 디버깅용 (필요 시 주석 해제)
        # if not detected:
        #    print(f"[{agent_name}] Lost {target_name}! Using Memory: {final_val}")
            
        return final_val[0], final_val[1], detected

    def _calculate_lidar_dynamics(self, agent, target, curr_x, curr_y):
        """
        [핵심] GT 없이 오직 Lidar의 (현재 위치 - 이전 위치) 차분으로 속도 정보 계산
        """
        # 1. 이전 위치 가져오기 (Reset 시 초기화 필수)
        if agent not in self.prev_lidar_pos: self.prev_lidar_pos[agent] = {}
        if target not in self.prev_lidar_pos[agent]: self.prev_lidar_pos[agent][target] = [curr_x, curr_y]
        
        prev_x, prev_y = self.prev_lidar_pos[agent][target]
        
        # 2. dt 체크 (0 방지)
        dt = self.dt if self.dt > 1e-6 else 0.01

        # 3. 상대 속도 벡터 추정 (Relative Velocity)
        # V_rel = (P_curr - P_prev) / dt
        vx_rel = (curr_x - prev_x) / dt
        vy_rel = (curr_y - prev_y) / dt

        # 4. 거리 및 벡터 연산
        R_vec = np.array([curr_x, curr_y])   # 상대 위치
        V_vec = np.array([vx_rel, vy_rel])   # 상대 속도 (추정)
        dist = np.linalg.norm(R_vec) + 1e-6

        # 5. PN 유도 변수 계산
        # Closing Speed: 가까워지면 +, 멀어지면 -
        closing_speed = -float(np.dot(R_vec, V_vec)) / dist
        
        # LOS Rate: 시선각 변화율
        cross_prod = float(R_vec[0]*V_vec[1] - R_vec[1]*V_vec[0])
        los_rate = cross_prod / (dist**2)
        
        # 6. 정규화 (학습용)
        # 30m/s, 10rad/s는 경험적 Max 값
        norm_closing = np.clip(closing_speed / 30.0, -1.0, 1.0)
        norm_los = np.clip(los_rate / 10.0, -1.0, 1.0)

        return norm_closing, norm_los
    
    def _get_current_location(self):
        self.current_location = {}
        self.current_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.current_location[agent] = self.client.simGetObjectPose(agent)
        self.current_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

    # ======================================================================
    # 초기화/이동/시각화 관련
    # ======================================================================
    def _hover(self, name):
        self.client.moveByVelocityZAsync(
            vx=0.0, vy=0.0,
            z=self.fixed_z,
            duration=0.3,
            vehicle_name=name
        ).join()

        try:
            self.client.hoverAsync(vehicle_name=name).join()
        except:
            pass

    def _setup_flight(self):
        self.client.enableApiControl(True, vehicle_name="Drone1")
        self.client.armDisarm(True, vehicle_name="Drone1")

        for agent in self.possible_agents:
            self.client.enableApiControl(True, vehicle_name=agent)
            self.client.armDisarm(True, vehicle_name=agent)

        self.client.enableApiControl(True, vehicle_name=self.dynamic_name)
        self.client.armDisarm(True, vehicle_name=self.dynamic_name)

        # 1. 이륙 (Takeoff) - 시동 걸기용
        cmds = []
        cmds.append(self.client.takeoffAsync(vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.takeoffAsync(vehicle_name=agent))
        cmds.append(self.client.takeoffAsync(vehicle_name=self.dynamic_name))

        for c in cmds:
            c.join()

        # 2. [수정] 이동(MoveTo) 대신 텔레포트(SetPose) 사용
        # 물리적 이동 없이 좌표를 강제로 찍어버려서 관성을 0으로 만듭니다.
        
        # (1) Leader 위치 고정
        pose_leader = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose_leader, True, vehicle_name="Drone1")
        
        # (2) Followers 위치 고정 (겹치지 않게 약간 분산시켜도 됨, 여기선 0,0으로 모음)
        # 만약 에이전트끼리 겹쳐서 튕겨나간다면 x, y에 약간의 오프셋을 주어야 합니다.
        for i, agent in enumerate(self.possible_agents):
            # 예: 에이전트들을 리더 주변에 약간 띄워서 배치하고 싶다면 아래 주석 해제
            # offset_x = (i % 2) * 2.0 - 1.0  
            # offset_y = (i // 2) * 2.0 - 1.0
            # pose_agent = airsim.Pose(airsim.Vector3r(offset_x, offset_y, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
            
            # 현재는 모두 0,0에 겹쳐도 물리 충돌 무시 설정이 되어있다면 괜찮지만, 보통은 띄우는게 좋습니다.
            # 일단 기존 코드 의도대로 0,0에 둡니다.
            pose_agent = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
            self.client.simSetVehiclePose(pose_agent, True, vehicle_name=agent)

        # (3) 적(DynamicObstacle) 위치 고정 (초기화는 나중에 _reset_obstacle_logic에서 하겠지만 안전상 고정)
        pose_enemy = airsim.Pose(airsim.Vector3r(0, 0, self.fixed_z), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose_enemy, True, vehicle_name=self.dynamic_name)

        # 3. [중요] 속도 0으로 강제 초기화 (Momentum Kill)
        # 위치를 옮겨도 이전 속도 벡터가 남아있을 수 있으므로 0으로 덮어씁니다.
        cmds = []
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name="Drone1"))
        for agent in self.possible_agents:
            cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=agent))
        cmds.append(self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name))
        
        for c in cmds:
            c.join()
            
        time.sleep(0.5) # 물리 엔진 안정화 대기

        # 4. 초기 위치 저장
        self.start_location["Drone1"] = self.client.simGetObjectPose("Drone1")
        for agent in self.agents:
            self.start_location[agent] = self.client.simGetObjectPose(agent)
        self.start_location[self.dynamic_name] = self.client.simGetObjectPose(self.dynamic_name)

        # Hover 함수 호출 (최종 안정화)
        self._hover("Drone1")
        for agent in self.possible_agents:
            self._hover(agent)
        self._hover(self.dynamic_name)

    def _update_leader_movement(self):
        # 1. 시각화 (기존 코드 유지)
        if self.do_visualize:
            now = time.time()
            if (now - self._last_visualize_t) >= 0.1:
                self.client.simFlushPersistentMarkers()
                self._visualize_circles()
                self._last_visualize_t = now

        # 2. 리더 이동 로직 (추가됨)
        # 방향 변경 타이머 체크
        if self.leader_move_timer <= 0:
            # 현재 위치 확인 (너무 멀리 갔는지 체크)
            pos = self.client.simGetObjectPose("Drone1").position
            dist_from_origin = math.hypot(pos.x_val, pos.y_val)

            if dist_from_origin > 50.0:
                # 50m 밖이면 원점(0,0) 쪽으로 방향 전환
                angle = math.atan2(-pos.y_val, -pos.x_val)
                # 약간의 랜덤성을 주어 자연스럽게 (-20도 ~ +20도)
                angle += random.uniform(-0.3, 0.3)
            else:
                # 안전 범위 내라면 완전 랜덤 방향
                angle = random.uniform(0, 2 * math.pi)

            vx = math.cos(angle) * self.REAL_LEADER_SPEED
            vy = math.sin(angle) * self.REAL_LEADER_SPEED
            
            self.current_leader_vel = [vx, vy]
            
            # 다음 방향 전환까지 유지할 스텝 수 (약 2~5초)
            self.leader_move_timer = random.randint(20, 50)
        
        self.leader_move_timer -= 1

        # AirSim에 속도 명령 전송
        # Yaw는 고정(0), 높이는 fixed_z 유지
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0.0)
        self.client.moveByVelocityZAsync(
            vx=self.current_leader_vel[0],
            vy=self.current_leader_vel[1],
            z=self.fixed_z,
            duration=self.dt, # step 시간만큼 이동
            yaw_mode=yaw_mode,
            vehicle_name="Drone1"
        ) # .join()을 쓰지 않아 비동기로 부드럽게 연결

    def _visualize_circles(self):
        try:
            leader_pos = self.client.simGetObjectPose("Drone1").position
            center = np.array([leader_pos.x_val, leader_pos.y_val, leader_pos.z_val], dtype=float)

            def ring_points(radius, n=36):
                pts = []
                for i in range(n + 1):
                    ang = (i / n) * 2 * np.pi
                    x = center[0] + radius * np.cos(ang)
                    y = center[1] + radius * np.sin(ang)
                    z = center[2]
                    pts.append(airsim.Vector3r(x, y, z))
                return pts

            line_thickness = 20.0
            self.client.simPlotLineStrip(
                ring_points(self.optimal_distance),
                [1, 1, 0, 0.8],
                line_thickness, 0.15, True
            )
            self.client.simPlotLineStrip(
                ring_points(self.far_cutoff),
                [0, 1, 0, 0.8],
                line_thickness, 0.15, True
            )
        except Exception:
            pass

    # ======================================================================
    # 보상/종료 관련
    # ======================================================================
    def _compute_reward(self, agent):
        # 1. 메모리에서 타겟(적)의 정규화된 위치 가져오기
        norm_pos = self.lidar_memory[agent][self.dynamic_name]
        
        # 2. 미터 단위로 복원 (동역학 계산용)
        curr_x_meter = norm_pos[0] * 100.0
        curr_y_meter = norm_pos[1] * 100.0
        
        # 3. 속도/각속도 계산 (인자: agent, target_name, x, y)
        c_speed, l_rate = self._calculate_lidar_dynamics(agent, self.dynamic_name, curr_x_meter, curr_y_meter)
        
        # 4. 보상 계산
        r_close = self.W_CLOSE * c_speed if c_speed > 0 else 0.0
        r_los = self.W_LOS * (1.0 - abs(l_rate))
        
        # return 값은 float 하나만 반환 (step 함수에서 unpacking 에러 방지)
        return float(r_close + r_los - self.STEP_PENALTY)

    def _eliminate_agent_only(self, agent_name):
        """아군끼리 충돌했을 때 해당 아군만 전장에서 제외."""
        if agent_name in self.eliminated_agents:
            return
        self.eliminated_agents.add(agent_name)
        far_pose = airsim.Pose(
            airsim.Vector3r(9999, 9999, -100),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        self.client.simSetVehiclePose(far_pose, True, vehicle_name=agent_name)
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=agent_name)

    def _end_episode(self, reward, status):
        _obs_list = []
        _rewards_list = []
        _terminations_list = []
        _infos_list = []

        # [수정 1] 성공/실패 판정 로직을 TTG와 동일하게 상세화
        is_success = 1.0 if status == "SUCCESS_DISTANCE_AGENT_DYNAMIC" else 0.0
        
        # 리더 피격 (적 -> 리더)
        is_leader_hit = 1.0 if "LEADER_AND_DYNAMIC" in status else 0.0
        
        # 아군 충돌 (에이전트끼리 OR 에이전트가 리더와 충돌)
        # PN 기존 로직은 "AGENT_AGENT"만 체크했으나, 리더와의 충돌도 아군 충돌로 보는 것이 맞음
        if self.episode_ally_collision or "AGENT_AGENT" in status or "AGENT_LEADER" in status:
            is_ally_collision = 1.0 
        else:
            is_ally_collision = 0.0
            
        # 이탈 (Far Cutoff)
        is_obj_collision = 1.0 if "FAR_CUTOFF" in status else 0.0
# --- [추가] 요격 거리(IR) 계산 ---
        intercept_range = 0.0
        if is_success:
            lx = self.current_location["Drone1"].position.x_val
            ly = self.current_location["Drone1"].position.y_val
            ex = self.current_location[self.dynamic_name].position.x_val
            ey = self.current_location[self.dynamic_name].position.y_val
            intercept_range = math.hypot(lx - ex, ly - ey)
        # --- [추가] 요격/종료 순간의 기하(Geometry) 배치 좌표 추출 ---
        final_positions = {
            "Leader": [self.current_location["Drone1"].position.x_val, 
                       self.current_location["Drone1"].position.y_val],
            "Enemy": [self.current_location[self.dynamic_name].position.x_val, 
                      self.current_location[self.dynamic_name].position.y_val],
            "Agents": [[self.current_location[a].position.x_val, 
                        self.current_location[a].position.y_val] for a in self.possible_agents]
        }

        self.stats_history["win"].append(is_success)
        self.stats_history["coll_leader"].append(is_leader_hit)
        self.stats_history["coll_drone"].append(is_ally_collision)
        self.stats_history["coll_obj"].append(is_obj_collision)

        def get_rate(key):
            if len(self.stats_history[key]) > 0:
                return sum(self.stats_history[key]) / len(self.stats_history[key])
            return 0.0

        win_rate = get_rate("win")

        for agent in self.possible_agents:
            _obs_list.append(self._get_obs(agent))
            _rewards_list.append(reward)
            _terminations_list.append(True)

            # [수정 2] info 구조를 { "AgentName": { ... } } 형태로 중첩 (TTG와 통일)
            # Runner가 이 구조를 인식하여 wandb 로그를 수집합니다.
            _infos_list.append({
                agent: {
                    "final_status": status,
                    "episode_success": is_success,
                    "episode_leader_hit": is_leader_hit,
                    "episode_ally_collision": is_ally_collision,
                    "win_rate": win_rate,
                    "cur_episode_steps": self.step_count,
                    # [추가] 새로 계산한 평가 지표 넘기기
                    "episode_intercept_range": intercept_range,
                    "primary_miss_occurred": self.primary_miss_occurred,
                    "final_positions": final_positions
                }
            })
        
        print(f"[{self.episode_count} Ep] WinRate: {win_rate:.2f} | Status: {status}")
        return _obs_list, _rewards_list, _terminations_list, _infos_list
    
    # --------------------- 동적장애물 FSM ---------------------
    def _update_dynamic_obstacle(self):
        self._obs_step_timer += 1
        target_speed = 10

        if self._obstacle_state == "IDLE":
            # IDLE 상태: 정지
            self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name)
            if self._obs_step_timer >= self._idle_wait_steps:
                self._obstacle_state = "ATTACK"
                self._obs_step_timer = 0

        elif self._obstacle_state == "ATTACK":
            try:
                # [수정] 이동하는 리더를 맞추기 위해 실시간 위치(Ground Truth)를 직접 조회
                leader_pose_live = self.client.simGetObjectPose("Drone1").position
                obs_pos_live = self.client.simGetObjectPose(self.dynamic_name).position

                l_vec = np.array([leader_pose_live.x_val, leader_pose_live.y_val, leader_pose_live.z_val])
                o_vec = np.array([obs_pos_live.x_val, obs_pos_live.y_val, obs_pos_live.z_val])

                diff = l_vec - o_vec
                dist = np.linalg.norm(diff)

                if dist > 0.5:
                    direction = diff / dist
                    vel = direction * target_speed

                    self.client.moveByVelocityAsync(
                        vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                        duration=0.1, # 짧게 갱신
                        vehicle_name=self.dynamic_name
                    )
            except Exception as e:
                print(f"Attack Logic Error: {e}")

    def _teleport_obstacle_randomly(self):
        leader_pos = self.client.simGetVehiclePose("Drone1").position
        lx, ly = leader_pos.x_val, leader_pos.y_val
        radius = 55.0
        angle = random.uniform(0, 2 * math.pi)

        tx = lx - 20 + radius * math.cos(angle)
        ty = ly - 20 + radius * math.sin(angle)
        tz = self.fixed_z

        pose = airsim.Pose(airsim.Vector3r(tx, ty, tz), airsim.Quaternionr(0, 0, 0, 1))
        self.client.simSetVehiclePose(pose, True, vehicle_name=self.dynamic_name)
        self.client.moveByVelocityAsync(0, 0, 0, duration=0.1, vehicle_name=self.dynamic_name).join()

    def _reset_obstacle_logic(self):
        self._teleport_obstacle_randomly()
        self._obstacle_state = "IDLE"
        self._obs_step_timer = 0
        self._idle_wait_steps = random.randint(10, 30)
        self.dynamic_obstacle_speed = random.uniform(self.obs_speed_min, self.obs_speed_max)
        print(f"[DynamicObstacle] Reset to IDLE. Waiting for {self._idle_wait_steps} steps.")

    def _check_distance_collision(self, name_a, name_b, threshold):
        pa = self.current_location[name_a].position
        pb = self.current_location[name_b].position

        dx = pa.x_val - pb.x_val
        dy = pa.y_val - pb.y_val

        dist = math.sqrt(dx * dx + dy * dy)
        return dist < threshold, dist

    # ======================================================================
    # RL/PettingZoo API
    # ======================================================================
    @property
    def observation_space(self):
        return [self.observation_spaces[a] for a in self.possible_agents]

    @property
    def action_space(self):
        return [self.action_spaces[a] for a in self.possible_agents]

    @property
    def share_observation_space(self):
        return [self.share_observation_spaces for _ in self.possible_agents]

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def _get_obs(self, agent):
        if agent in self.eliminated_agents:
            return np.zeros(self.observation_spaces[agent].shape, dtype=np.float32)

        # 1. 내 상태 (속도) - IMU/GPS는 자신의 정보이므로 그대로 사용
        try:
            my_state = self.client.getMultirotorState(vehicle_name=agent)
            vx = my_state.kinematics_estimated.linear_velocity.x_val
            vy = my_state.kinematics_estimated.linear_velocity.y_val
            norm_vx = np.clip(vx / self.MAX_SPEED, -1.0, 1.0)
            norm_vy = np.clip(vy / self.MAX_SPEED, -1.0, 1.0)
        except:
            norm_vx, norm_vy = 0.0, 0.0

        # 2. 리더 위치 (Lidar)
        lx, ly, _ = self._get_lidar_measurement(agent, "Drone1")
        _leader_feats = [lx, ly]

        # 3. 아군 위치 (Lidar) - 고정 반복문
        _ally_feats = []
        other_agents = [a for a in self.possible_agents if a != agent]
        for other in other_agents:
            ox, oy, _ = self._get_lidar_measurement(agent, other)
            _ally_feats.append(ox)
            _ally_feats.append(oy)

        # 4. 적군 위치 및 동적 정보 (Pure Lidar Logic)
        # Lidar 위치 기반 미분으로 변경
        
        # (1) Lidar 관측 (정규화된 값: -1.0 ~ 1.0)
        ex, ey, is_enemy_visible = self._get_lidar_measurement(agent, self.dynamic_name)
        
        # (2) 정규화 해제 (Dynamics 계산을 위해 미터 단위로 변환)
        # 가정: Lidar 정규화 상수가 100.0m라고 할 때
        ex_meter = ex * 100.0
        ey_meter = ey * 100.0
        
        # (3) Lidar 히스토리 기반 속도 정보 계산
        # _calculate_lidar_dynamics는 (현재-과거)/dt 를 수행하므로 미터 단위가 필요함
        # 리턴값(closing_norm, los_norm)은 이미 함수 내부에서 정규화되어 나옴
        if is_enemy_visible:
            closing_norm, los_norm = self._calculate_lidar_dynamics(agent, self.dynamic_name, ex_meter, ey_meter)
        else:
            # 적을 놓쳤을 경우: 속도 정보를 0으로 두거나, 메모리에 있는 마지막 속도를 유지할 수도 있음.
            # 여기서는 안전하게 0으로 처리 (혹은 이전 프레임 값을 유지하려면 변수 따로 관리 필요)
            closing_norm, los_norm = 0.0, 0.0

        _dynamic_feats = [
            ex, ey,         # Lidar 위치 (Normalized)
            closing_norm,   # Lidar 기반 계산된 접근 속도 (Normalized)
            los_norm        # Lidar 기반 계산된 LOS Rate (Normalized)
        ]

        # 5. 최종 결합
        obs = np.concatenate([
            np.array(_leader_feats, dtype=np.float32).flatten(),
            np.array(_ally_feats, dtype=np.float32).flatten(),
            np.array(_dynamic_feats, dtype=np.float32).flatten(),
            np.array([norm_vx, norm_vy], dtype=np.float32) # 내 속도
        ]).astype(np.float32)

        return obs
    

    def _do_action(self, actions):
        actions = np.clip(actions, -1, 1)
        dt = self.dt

        for i, agent in enumerate(self.possible_agents):
            if agent in self.eliminated_agents:
                continue
            a = actions[i]
            # [2D UFO Mode]
            # a[0]: 전후 (Global X), a[1]: 좌우 (Global Y)
            # YawRate: 0 (Fixed)
            
            v_forward = float(a[0]) * self.MAX_SPEED  
            v_lateral = float(a[1]) * self.MAX_SPEED

            sp = math.hypot(v_forward, v_lateral)
            if sp > self.MAX_SPEED:
                s = self.MAX_SPEED / (sp + 1e-6)
                v_forward *= s
                v_lateral *= s

            vx = v_forward 
            vy = v_lateral 

            yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=0.0)

            self.client.moveByVelocityZAsync(
                vx=vx, vy=vy, z=self.fixed_z, duration=dt,
                yaw_mode=yaw_mode,
                vehicle_name=agent
            )

    def _get_rewards(self, per_agent_results):
        return [np.mean(per_agent_results) for _ in self.possible_agents]

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        print(f"에피소드: {self.episode_count} | 소비한 스텝 수: {self.step_count}")

        self.step_count = 0
        self.agents = self.possible_agents[:]
        self.eliminated_agents.clear()
        self.episode_ally_collision = False

        self.leader_move_timer = 0
        self.current_leader_vel = [0,0, 0.0]

        self.client.reset()
        self._setup_flight()
        self.client.simFlushPersistentMarkers()

        self._reset_obstacle_logic()
        self._get_current_location()

        self._prev_d_agent_enemy = {a: None for a in self.possible_agents}
        self._last_pose.clear()
        self._last_time.clear()

        self._last_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}
        self._current_action = {a: np.zeros(3, dtype=np.float32) for a in self.possible_agents}

        # 메모리 및 이전 위치 초기화
        self.lidar_memory = {}
        self.prev_lidar_pos = {}
        
        for agent in self.possible_agents:
            self.lidar_memory[agent] = {}
            self.prev_lidar_pos[agent] = {}
            
            # 초기엔 타겟 위치 모름 (100m 끝지점으로 가정)
            # 여기서는 정규화 전인 '실제 미터' 단위로 저장하는 것이 계산에 편합니다.
            default_real_pos = [100.0, 100.0] 
            
            self.lidar_memory[agent]["Drone1"] = [1.0, 1.0] # Obs용 정규화 값
            self.prev_lidar_pos[agent]["Drone1"] = default_real_pos # 속도계산용 실제 값

            self.lidar_memory[agent][self.dynamic_name] = [1.0, 1.0]
            self.prev_lidar_pos[agent][self.dynamic_name] = default_real_pos
            
            # 아군들
            for other in self.possible_agents:
                if agent != other:
                    self.lidar_memory[agent][other] = [1.0, 1.0]
                    self.prev_lidar_pos[agent][other] = default_real_pos

        self.primary_miss_occurred = False
        self._prev_dist_to_enemy = {a: 999.0 for a in self.possible_agents}

        obs_list = [self._get_obs(a) for a in self.agents]
        return obs_list

    def step(self, actions):
        self.step_count += 1

        if self.step_count >= 300:
            print(f"⏳[시간 초과] 스텝 {self.step_count} 도달! → 실패 처리")
            return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_TIMEOUT_LEADER_HIT")

        self._do_action(actions)
        # self._update_leader_movement()
        self._update_dynamic_obstacle()
        self._get_current_location()

        # 먼저 리더 피격 확인: 모든 아군이 제거되더라도 반드시 검사한다.
        hit_L, d_L = self._check_distance_collision(
            "Drone1", self.dynamic_name, self.COLLISION_THRESHOLD
        )
        if hit_L:
            return self._end_episode(self.REWARD_LEADER_HIT, "FAIL_LEADER_AND_DYNAMIC_DISTANCE")

        # 활성 아군의 리더와 충돌 / 임무구역 이탈 확인.
        active = [a for a in self.possible_agents if a not in self.eliminated_agents]
        for agent in active:
            ap = self.current_location[agent].position
            lp = self.current_location["Drone1"].position
            d_leader = np.linalg.norm([
                ap.x_val - lp.x_val, ap.y_val - lp.y_val, ap.z_val - lp.z_val
            ])
            if d_leader > self.far_cutoff:
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_AGENT_FAR_CUTOFF")
            hit, _ = self._check_distance_collision(agent, "Drone1", self.COLLISION_THRESHOLD)
            if hit:
                return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_DISTANCE_AGENT_LEADER")

        # 아군-아군 충돌: 에피소드 종료 대신 충돌 당사자만 제거한다.
        collided = set()
        for i, agent in enumerate(active):
            if agent in collided:
                continue
            for other in active[i + 1:]:
                if other in collided:
                    continue
                hit, dist = self._check_distance_collision(agent, other, self.COLLISION_THRESHOLD)
                if hit:
                    print(f"💥🤖[아군 충돌] {agent} ↔ {other} (d={dist:.2f}m): 두 대 제거 후 계속 진행")
                    collided.update((agent, other))
                    break

        if collided:
            self.episode_ally_collision = True
            for agent in collided:
                self._eliminate_agent_only(agent)

        active = [a for a in self.possible_agents if a not in self.eliminated_agents]
        if not active:
            return self._end_episode(self.REWARD_AGENT_CRASH, "FAIL_ALL_ALLIES_ELIMINATED")

        # 남은 아군의 요격 성공 / Miss 판정.
        for agent in active:
            ep = self.current_location[self.dynamic_name].position
            ap = self.current_location[agent].position
            curr_dist = math.hypot(ep.x_val - ap.x_val, ep.y_val - ap.y_val)
            if (curr_dist > self._prev_dist_to_enemy[agent]
                    and self._prev_dist_to_enemy[agent] < self.MISS_THRESHOLD
                    and self._prev_dist_to_enemy[agent] > self.COLLISION_THRESHOLD):
                self.primary_miss_occurred = True
            self._prev_dist_to_enemy[agent] = curr_dist

            hit, _ = self._check_distance_collision(agent, self.dynamic_name, self.COLLISION_THRESHOLD)
            if hit:
                print(f"🎯🔥[요격] {agent}가 적을 요격 성공!")
                return self._end_episode(self.REWARD_HIT_ENEMY, "SUCCESS_DISTANCE_AGENT_DYNAMIC")

        obs_list = []
        per_agent_results = []
        infos_list = []
        for agent in self.possible_agents:
            if agent in self.eliminated_agents:
                obs_list.append(self._get_obs(agent))  # zero observation
                per_agent_results.append(0.0)
                infos_list.append({"step_reward": 0.0, "eliminated": True})
                continue
            obs_list.append(self._get_obs(agent))
            reward = self._compute_reward(agent)
            per_agent_results.append(reward)
            infos_list.append({"step_reward": reward, "eliminated": False})
            norm_curr = self.lidar_memory[agent][self.dynamic_name]
            self.prev_lidar_pos[agent][self.dynamic_name] = [norm_curr[0] * 100.0, norm_curr[1] * 100.0]

        # 이전 코드의 아군 충돌 패널티 -500을 유지하되, 이 스텝에서만 적용.
        # 기존 정책의 팀 공유 보상 구조를 유지한다.
        rewards_list = ([float(self.REWARD_AGENT_CRASH)] * len(self.possible_agents)
                        if collided else self._get_rewards(per_agent_results))
        termination_list = [False] * len(self.possible_agents)
        return obs_list, rewards_list, termination_list, infos_list
