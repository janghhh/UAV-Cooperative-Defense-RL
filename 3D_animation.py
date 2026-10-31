import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

JSON_PATH = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\궤적\20m\episode_033.json"
SAVE_PATH = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\궤적\30.gif"

TAIL_LENGTH = 40
INTERVAL_MS = 100

with open(JSON_PATH, "r", encoding="utf-8") as f:
    episode_data = json.load(f)

trace = episode_data.get("trace", [])

if not trace:
    raise ValueError("trace가 비어 있습니다.")


# ==========================================================
# 좌표 형식 통합 함수
# - 형식 1: {"x": 1.0, "y": 2.0, "z": -10.0}
# - 형식 2: [1.0, 2.0, -10.0]
# 둘 다 처리 가능
# ==========================================================
def get_xyz(pos):
    if pos is None:
        return None

    # dict 형식: {"x": ..., "y": ..., "z": ...}
    if isinstance(pos, dict):
        x = pos.get("x", None)
        y = pos.get("y", None)
        z = pos.get("z", 0.0)

        if x is None or y is None:
            return None

        return float(x), float(y), float(z)

    # list 또는 tuple 형식: [x, y, z]
    if isinstance(pos, (list, tuple)):
        if len(pos) < 2:
            return None

        x = pos[0]
        y = pos[1]
        z = pos[2] if len(pos) >= 3 else 0.0

        return float(x), float(y), float(z)

    return None


# ==========================================================
# 혹시 positions가 없는 구형 trace 구조도 방어
# leader / agents / enemies 구조를 positions로 변환
# ==========================================================
for fr in trace:
    if "positions" not in fr:
        positions = {}

        if "leader" in fr and fr["leader"] is not None:
            positions["Drone1"] = fr["leader"]

        if "agents" in fr and isinstance(fr["agents"], dict):
            positions.update(fr["agents"])

        if "enemies" in fr and isinstance(fr["enemies"], dict):
            positions.update(fr["enemies"])

        fr["positions"] = positions


# ==========================================================
# 전체 객체 이름 수집
# ==========================================================
all_names = set()

for fr in trace:
    positions = fr.get("positions", {})
    all_names.update(positions.keys())

all_names = sorted(all_names)

if len(all_names) == 0:
    raise ValueError("positions 안에 객체 좌표가 없습니다.")


# ==========================================================
# 색상 함수
# - Drone1: gold
# - DynamicObstacle: red
# - Follower 기본: blue
# - 해당 frame에서 interceptor인 Follower만 green
#
# 중요:
# final_interceptor를 사용하지 않음.
# 즉, 최종 요격자를 처음부터 미리 초록색으로 칠하지 않음.
# ==========================================================
def get_color(name, frame_idx=None):
    # 리더
    if name == "Drone1":
        return "gold"

    # 적기
    if name.startswith("DynamicObstacle"):
        return "red"

    # Follower가 아닌 기타 객체
    if not name.startswith("Follower"):
        return "gray"

    # frame 정보가 없으면 Follower는 기본 파란색
    if frame_idx is None:
        return "blue"

    fr = trace[frame_idx]

    # 1. interceptor_map 기준
    # 예: {"DynamicObstacle0": "Follower1"}
    interceptor_map = fr.get("interceptor_map", {})

    if isinstance(interceptor_map, dict):
        if name in interceptor_map.values():
            return "green"

    # 2. current_interceptors 기준
    # 예: ["Follower1"]
    current_interceptors = fr.get("current_interceptors", [])

    if isinstance(current_interceptors, list):
        if name in current_interceptors:
            return "green"

    # 3. roles 기준
    # 예: {"Follower0": "escort", "Follower1": "interceptor"}
    roles = fr.get("roles", {})

    if isinstance(roles, dict):
        if roles.get(name, None) == "interceptor":
            return "green"

    # 현재 frame에서 요격자가 아니면 파란색
    return "blue"


# ==========================================================
# 전체 x, y 범위 계산
# ==========================================================
xs, ys = [], []

for fr in trace:
    positions = fr.get("positions", {})

    for pos in positions.values():
        xyz = get_xyz(pos)

        if xyz is None:
            continue

        x, y, z = xyz
        xs.append(x)
        ys.append(y)

if len(xs) == 0 or len(ys) == 0:
    raise ValueError("유효한 x, y 좌표가 없습니다. JSON의 positions 구조를 확인하세요.")

xmin, xmax = min(xs), max(xs)
ymin, ymax = min(ys), max(ys)

pad = 3.0
xmin, xmax = xmin - pad, xmax + pad
ymin, ymax = ymin - pad, ymax + pad


# ==========================================================
# Figure 생성
# ==========================================================
fig, ax = plt.subplots(figsize=(8, 8))


def update(frame_idx):
    ax.clear()

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    # 제목/축 이름 제거
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # 일정 간격 grid line 유지
    tick_step = 10

    xticks = np.arange(
        np.floor(xmin / tick_step) * tick_step,
        np.ceil(xmax / tick_step) * tick_step + tick_step,
        tick_step
    )

    yticks = np.arange(
        np.floor(ymin / tick_step) * tick_step,
        np.ceil(ymax / tick_step) * tick_step + tick_step,
        tick_step
    )

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # 숫자는 숨기고 grid와 박스는 유지
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    ax.grid(True, color="lightgray", alpha=0.6, linewidth=0.8)

    # 네모 박스 유지
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")

    start_idx = max(0, frame_idx - TAIL_LENGTH)

    for name in all_names:
        tx, ty = [], []

        # 꼬리 궤적
        for fr in trace[start_idx:frame_idx + 1]:
            pos = fr.get("positions", {}).get(name, None)
            xyz = get_xyz(pos)

            if xyz is None:
                continue

            x, y, z = xyz
            tx.append(x)
            ty.append(y)

        # 현재 frame 기준 색상
        color = get_color(name, frame_idx)

        if len(tx) > 1:
            ax.plot(tx, ty, color=color, linewidth=2)

        # 현재 위치
        cur = trace[frame_idx].get("positions", {}).get(name, None)
        cur_xyz = get_xyz(cur)

        if cur_xyz is not None:
            x, y, z = cur_xyz
            ax.scatter(x, y, color=color, s=60)
            ax.text(x, y, f" {name}", fontsize=9, va="center")


ani = FuncAnimation(
    fig,
    update,
    frames=len(trace),
    interval=INTERVAL_MS,
    repeat=False
)

writer = PillowWriter(fps=10)
ani.save(SAVE_PATH, writer=writer, dpi=120)

print(f"2D GIF 저장 완료: {SAVE_PATH}")

plt.show()