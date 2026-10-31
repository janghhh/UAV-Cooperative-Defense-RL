# ---- UTF-8 safe console (Windows cp949 crash fix) ----
import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import os
import torch
import numpy as np
import gym
from gym import spaces
from types import SimpleNamespace
import wandb
import json

# === 프로젝트 모듈 ===
from onpolicy.envs.eval_env import AirSimMultiDroneEnv
# from onpolicy.envs.eval_env_2 import AirSimMultiDroneEnv
# from onpolicy.envs.eval_env_3 import AirSimMultiDroneEnv
# from onpolicy.envs.make_env_PN_Lidar_hybrid_guardian_curriculum import AirSimMultiDroneEnv
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

# ---------- [설정] 경로 및 디바이스 ----------
# actor_model_path = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260713_124648-yrd8tntb\files\actor.pt" #5대
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\40m\wandb\run-20260610_173403-v1g5a6lr\files\actor.pt" #4대
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260716_131521-fbj2kxgv\files\actor.pt" # 4대
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\40m\wandb\run-20260610_173403-v1g5a6lr\files\actor.pt" # 4대
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\하이브리드\wandb\run-20260813_091338-wsyvarl9\files\actor.pt"
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260902_112750-8tcajpbs\files\actor.pt" # 4대
actor_model_path = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260826_132219-v3slzkyx\files\actor.pt" # 7대

# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\하이브리드\wandb\run-20260420_113517-7guao0iy\files\actor.pt"# 동적 50m
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\40m\wandb\run-20260518_122019-2k1i6igh\files\actor.pt" # 동적 40m
# actor_model_path = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260727_093412-amvc4uwo\files\actor.pt" # 동적 30m
# actor_model_path = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260724_104644-9w31ras9\files\actor.pt" # 동적 30m
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\40m\wandb\run-20260521_180339-4lfazm51\files\actor.pt" # 동적 20m
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\하이브리드\wandb\run-20260504_100459-gcpiyby3\files\actor.pt" #정적 50m
# actor_model_path = r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\커리큘럼 제외\wandb\run-20260813_091500-5u1w67z6\files\actor.pt" #정적 30m
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\동적역할\40m\wandb\run-20260528_115946-3le1oayg\files\actor.pt" # 동일 40m
# actor_model_path = r"C:\Users\rl_rnv_pc_3\Documents\Gardian\UAV_Airsim_RL_Project\UAV_Airsim_RL_Project\on-policy\하이브리드\wandb\run-20260427_150055-syf1ik0m\files\actor.pt" # 동일 50m

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ---------- [설정] 하이퍼파라미터 ----------
args = SimpleNamespace(
    hidden_size=64,
    gain=0.01,
    use_orthogonal=True,
    use_policy_active_masks=False,
    use_naive_recurrent_policy=False,
    use_recurrent_policy=True,
    recurrent_N=1,
    algorithm_name="rmappo",
    use_popart=False,
    use_feature_normalization=True,
    use_cnn_feature=False,
    stacked_frames=1,
    activation_id=1,
    use_ReLU=True,
    layer_N=1,
)

# ---------- [설정] 반복 평가 ----------
# 한 회차에서 평가할 에피소드 수
EPISODES_PER_RUN = 100

# 100에피소드 평가를 반복할 횟수
# 예: 10, 50, 100 등 원하는 값으로 변경
NUM_EVALUATION_RUNS = 50

# 회차별 궤적과 최종 요약 파일이 저장되는 기본 경로
BASE_SAVE_DIR = (
    r"C:\Users\RL_RNV_PC_3\Documents\Gardian\UAV_Airsim_RL_Project"
    r"\UAV_Airsim_RL_Project\on-policy\궤적\30m-다시"
)


def _get_first_space(env, attr):
    if hasattr(env, attr):
        sp = getattr(env, attr)
        if isinstance(sp, (list, tuple)):
            return sp[0]
        return sp
    return None


def main():
    # 1. WandB 초기화
    wandb_config = vars(args).copy()
    wandb_config.update({
        "episodes_per_run": EPISODES_PER_RUN,
        "num_evaluation_runs": NUM_EVALUATION_RUNS,
        "total_evaluation_episodes": EPISODES_PER_RUN * NUM_EVALUATION_RUNS,
    })

    wandb.init(
        project="UAV_Comparison",
        name=(
            f"Proposed_Method_Eval_"
            f"{EPISODES_PER_RUN}EP_x_{NUM_EVALUATION_RUNS}Runs"
        ),
        config=wandb_config,
        job_type="evaluation",
        reinit=True
    )

    # 2. Env 생성
    print("Initialize Environment...")
    env = AirSimMultiDroneEnv(
        ip_address="127.0.0.1",
        follower_names=("Follower0", "Follower1", "Follower2", "Follower3", "Follower4", "Follower5", "Follower6"),
        do_visualize=True
    )

    obs_list = env.reset()

    # 3. 모델 로드
    obs_space = _get_first_space(env, "observation_space")
    act_space = _get_first_space(env, "action_space")

    if obs_space is None:
        obs_dim = int(np.asarray(obs_list[0]).size)
        obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    if act_space is None:
        act_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

    actor = R_Actor(args, obs_space, act_space, device=device)

    if os.path.exists(actor_model_path):
        state = torch.load(actor_model_path, map_location=device)
        state_dict = state.get("model", state) if isinstance(state, dict) else state
        actor.load_state_dict(state_dict, strict=True)
        actor.eval()
        print("✅ Model Loaded Successfully!")
    else:
        print(f"❌ Error: Model file not found at {actor_model_path}")
        wandb.finish()
        return

    # 4. 100에피소드 평가를 지정한 횟수만큼 반복
    all_run_reports = []

    try:
        for run_index in range(1, NUM_EVALUATION_RUNS + 1):
            print("\n" + "#" * 80)
            print(
                f"🔁 Evaluation Run {run_index}/{NUM_EVALUATION_RUNS} 시작 "
                f"({EPISODES_PER_RUN} Episodes)"
            )
            print("#" * 80)

            report = evaluate_policy(
                env=env,
                actor=actor,
                num_episodes=EPISODES_PER_RUN,
                deterministic=False,
                run_index=run_index,
                total_runs=NUM_EVALUATION_RUNS,
                global_episode_offset=(run_index - 1) * EPISODES_PER_RUN,
            )
            all_run_reports.append(report)

        # 5. 모든 회차가 끝난 뒤, 회차별 결과만 처음부터 다시 출력
        print_all_evaluation_reports(all_run_reports)

    finally:
        # 환경에 close()가 구현되어 있는 경우 안전하게 종료
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()
        wandb.finish()

def evaluate_policy(
    env,
    actor,
    num_episodes=100,
    deterministic=False,
    run_index=1,
    total_runs=1,
    global_episode_offset=0,
):
    # --- [데이터 수집용 리스트] ---
    results = {
        "success": [],
        "ally_collision": [],
        "leader_collision": [],  # 리더기 충돌률
        "intercept_range": [],
        "primary_miss": [],
        "secondary_success": [],
        "intercept_positions": []
    }

    all_episode_traces = []

    print(
        f"🚀 Start Evaluation Run {run_index}/{total_runs} "
        f"for {num_episodes} episodes..."
    )

    # 회차별 폴더 분리: 다음 회차가 이전 회차의 JSON 파일을 덮어쓰지 않음
    save_dir = os.path.join(BASE_SAVE_DIR, f"run_{run_index:03d}")
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs_list = env.reset()
        done_list = [False] * len(obs_list)

        rnn_states = torch.zeros(
            (len(obs_list), args.recurrent_N, args.hidden_size),
            dtype=torch.float32,
            device=device
        )
        masks = torch.ones(
            (len(obs_list), 1),
            dtype=torch.float32,
            device=device
        )

        episode_done = False

        while not episode_done:
            obs_tensor = torch.tensor(
                np.array(obs_list),
                dtype=torch.float32,
                device=device
            )

            with torch.no_grad():
                act_t, _, rs_out = actor(
                    obs_tensor,
                    rnn_states,
                    masks,
                    available_actions=None,
                    deterministic=deterministic
                )

            actions = act_t.cpu().numpy()
            rnn_states = rs_out

            next_obs_list, rewards, done_list, info_list = env.step(actions)

            masks = torch.tensor(
                1 - np.array(done_list)[:, None],
                dtype=torch.float32,
                device=device
            )

            obs_list = next_obs_list

            if np.any(done_list):
                episode_done = True

                # --- [정보 추출] ---
                # 모든 에이전트의 info는 공유되므로 첫 번째만 사용
                first_info = list(info_list[0].values())[0]

                is_win = first_info.get("episode_success", 0)
                status = first_info.get("final_status", "")

                # 아군 충돌률
                is_coll = first_info.get("episode_ally_collision", 0)

                # 적기-리더기 충돌률
                # 리더기 충돌률
                is_leader_collision = first_info.get(
                    "episode_leader_hit",
                    1.0 if (
                        "LEADER_AND_DYNAMIC" in status
                        or "AGENT_LEADER" in status
                    ) else 0.0
                )

                ir = first_info.get("episode_intercept_range", 0.0)
                miss_occurred = first_info.get("primary_miss_occurred", False)
                interceptor = first_info.get("interceptor", None)

                trace = first_info.get("episode_trace", [])

                final_interceptor = first_info.get("final_interceptor", None)
                final_intercept_enemy = first_info.get("final_intercept_enemy", None)
                final_interceptor_role = first_info.get("final_interceptor_role", None)
                assigned_interceptor_before_hit = first_info.get(
                    "assigned_interceptor_before_hit",
                    None
                )

                # 혹시 info에는 없고 trace 마지막에만 있을 경우 fallback
                if final_interceptor is None:
                    for fr in reversed(trace):
                        if fr.get("final_interceptor", None) is not None:
                            final_interceptor = fr.get("final_interceptor", None)
                            final_intercept_enemy = fr.get("final_intercept_enemy", None)
                            final_interceptor_role = fr.get("final_interceptor_role", None)
                            assigned_interceptor_before_hit = fr.get(
                                "assigned_interceptor_before_hit",
                                None
                            )
                            break

                global_episode = global_episode_offset + ep + 1

                latest_episode_data = {
                    "evaluation_run": run_index,
                    "episode": ep + 1,
                    "global_episode": global_episode,
                    "success": bool(is_win),
                    "status": status,

                    "ally_collision": bool(is_coll),
                    "leader_collision": bool(is_leader_collision),

                    "interceptor": final_interceptor,
                    "final_interceptor": final_interceptor,
                    "final_intercept_enemy": final_intercept_enemy,
                    "final_interceptor_role": final_interceptor_role,
                    "assigned_interceptor_before_hit": assigned_interceptor_before_hit,

                    "trace": trace
                }

                all_episode_traces.append(latest_episode_data)

                # 1) 가장 최근 에피소드 1개 저장
                latest_episode_path = os.path.join(save_dir, "latest_episode_trace.json")
                with open(latest_episode_path, "w", encoding="utf-8") as f:
                    json.dump(latest_episode_data, f, indent=2, ensure_ascii=False)

                # 2) 에피소드별 개별 파일 저장
                episode_file_path = os.path.join(save_dir, f"episode_{ep+1:03d}.json")
                with open(episode_file_path, "w", encoding="utf-8") as f:
                    json.dump(latest_episode_data, f, indent=2, ensure_ascii=False)

                # 3) 전체 누적 저장
                all_episode_path = os.path.join(save_dir, "episode_trajectories.json")
                with open(all_episode_path, "w", encoding="utf-8") as f:
                    json.dump(all_episode_traces, f, indent=2, ensure_ascii=False)

                print(f"💾 Episode {ep+1} 저장 완료: {episode_file_path}")

                # 리스트에 저장
                results["success"].append(is_win)
                results["ally_collision"].append(is_coll)
                results["leader_collision"].append(is_leader_collision)

                # 요격 성공 시에만 IR 기록
                if is_win == 1.0:
                    results["intercept_range"].append(ir)
                    results["intercept_positions"].append(
                        first_info.get("final_positions", {})
                    )

                results["primary_miss"].append(miss_occurred)

                # 1차 요격 실패가 있었는데 최종적으로 성공했다면 → 차순위 요격 성공
                is_secondary_success = (miss_occurred and is_win == 1.0)
                results["secondary_success"].append(is_secondary_success)

                # WandB Step Log: 전체 반복 평가에서 step이 계속 증가
                wandb.log({
                    "Evaluation_Run": run_index,
                    "Episode_In_Run": ep + 1,
                    "Global_Episode": global_episode,
                    "Success": is_win,
                    "Ally_Collision": is_coll,
                    "Leader_Collision": is_leader_collision,
                    "Intercept_Range": ir if is_win else 0,
                    "Primary_Miss": int(miss_occurred)
                }, step=global_episode)

                print(
                    f"[Run {run_index:03d}/{total_runs:03d} | "
                    f"Ep {ep+1:03d}/{num_episodes} | "
                    f"Global Ep {global_episode:05d}] "
                    f"Win: {is_win} | "
                    f"AllyColl: {is_coll} | "
                    f"LeaderColl: {is_leader_collision} | "
                    f"IR: {ir:.2f}m | "
                    f"Missed Once?: {miss_occurred}"
                )

    # --- [해당 회차 결과 집계] ---
    actual_eps = len(results["success"])

    success_count = int(sum(results["success"]))
    ally_collision_count = int(sum(results["ally_collision"]))
    leader_collision_count = int(sum(results["leader_collision"]))

    # 1. DSR: 방어 성공률
    avg_dsr = (
        success_count / actual_eps * 100
        if actual_eps > 0 else 0.0
    )

    # 2. CR: 아군 충돌률
    avg_cr = (
        ally_collision_count / actual_eps * 100
        if actual_eps > 0 else 0.0
    )

    # 3. ELCR: 적기-리더기 충돌률
    avg_lcr = (
        leader_collision_count / actual_eps * 100
        if actual_eps > 0 else 0.0
    )

    # 4. IR: 요격 거리, 성공한 케이스만 평균 계산
    intercept_ranges = [float(v) for v in results["intercept_range"]]
    avg_ir = float(np.mean(intercept_ranges)) if intercept_ranges else 0.0

    # 5. SIR: 차순위 요격 성공률
    total_misses = int(sum(results["primary_miss"]))
    total_sec_success = int(sum(results["secondary_success"]))
    avg_sir = (
        total_sec_success / total_misses * 100
        if total_misses > 0 else 0.0
    )

    report = {
        "run_index": int(run_index),
        "num_episodes_requested": int(num_episodes),
        "actual_episodes": int(actual_eps),
        "success_count": success_count,
        "ally_collision_count": ally_collision_count,
        "leader_collision_count": leader_collision_count,
        "intercept_success_count": len(intercept_ranges),
        "primary_miss_count": total_misses,
        "secondary_success_count": total_sec_success,
        "DSR": float(avg_dsr),
        "CR": float(avg_cr),
        "LCR": float(avg_lcr),
        "IR_mean": float(avg_ir),
        "SIR": float(avg_sir),
        "save_dir": save_dir,
    }

    # 현재 회차 결과는 기존처럼 즉시 출력
    print_run_report(report, repeated_at_end=False)

    # 회차별 결과 JSON 저장
    run_report_path = os.path.join(save_dir, "evaluation_report.json")
    with open(run_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 회차별 시각화용 좌표 데이터 저장
    intercept_geometry_path = os.path.join(save_dir, "intercept_geometry.json")
    with open(intercept_geometry_path, "w", encoding="utf-8") as f:
        json.dump(
            results["intercept_positions"],
            f,
            indent=4,
            ensure_ascii=False
        )

    # WandB에는 회차 단위 결과도 별도로 기록
    wandb.log({
        "Run_Summary/Run": run_index,
        "Run_Summary/DSR": avg_dsr,
        "Run_Summary/CR": avg_cr,
        "Run_Summary/LCR": avg_lcr,
        "Run_Summary/IR_mean": avg_ir,
        "Run_Summary/SIR": avg_sir,
    }, step=global_episode_offset + actual_eps)

    print(f"💾 Run {run_index:03d} 결과 저장: {run_report_path}")
    print(f"💾 요격 좌표 저장: {intercept_geometry_path}")
    print(f"Evaluation Run {run_index}/{total_runs} Finished.")

    return report


def print_run_report(report, repeated_at_end=False):
    """한 평가 회차의 결과를 동일한 형식으로 출력한다."""
    run_index = report["run_index"]
    actual_eps = report["actual_episodes"]

    if repeated_at_end:
        title = f"📌 Run {run_index:03d} 결과 재출력"
    else:
        title = f"📊 Evaluation Run {run_index:03d} Report"

    print("\n" + "=" * 72)
    print(f"{title} (Total {actual_eps} Episodes)")
    print("=" * 72)
    print(
        f"● 방어 성공률 (DSR)           : {report['DSR']:.2f}% "
        f"({report['success_count']}/{actual_eps})"
    )
    print(
        f"● 아군 충돌률 (CR)            : {report['CR']:.2f}% "
        f"({report['ally_collision_count']}/{actual_eps})"
    )
    print(
        f"● 리더기 충돌률 (LCR)         : {report['LCR']:.2f}% "
        f"({report['leader_collision_count']}/{actual_eps})"
    )
    print(
        f"● 평균 요격 거리 (IR)         : {report['IR_mean']:.2f} m "
        f"*성공 에피소드 기준"
    )

    if report["primary_miss_count"] > 0:
        print(
            f"● 차순위 요격 성공률(SIR)      : {report['SIR']:.2f}% "
            f"({report['secondary_success_count']}회 성공 / "
            f"{report['primary_miss_count']}회 1차 실패 발생)"
        )
    else:
        print(
            "● 차순위 요격 성공률(SIR)      : "
            "N/A (1차 요격 실패가 단 한 번도 발생하지 않음)"
        )

    print("=" * 72)


def print_all_evaluation_reports(all_run_reports):
    """모든 평가 종료 후 각 회차 결과만 처음부터 다시 출력한다."""
    if not all_run_reports:
        print("⚠️ 출력할 평가 결과가 없습니다.")
        return

    print("\n\n" + "#" * 90)
    print("📚 모든 반복 평가 완료: 회차별 결과 전체 재출력")
    print("#" * 90)

    for report in all_run_reports:
        print_run_report(report, repeated_at_end=True)

    print("\n" + "#" * 90)
    print(f"✅ 총 {len(all_run_reports)}개 평가 회차 결과 재출력 완료")
    print("#" * 90)


if __name__ == "__main__":
    main()