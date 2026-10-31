# Cooperative Multi-UAV Defense Using Multi-Agent Reinforcement Learning

Research code for AirSim-based cooperative UAV defense experiments using recurrent multi-agent proximal policy optimization (rMAPPO), role-based rewards, dynamic role switching, and curriculum learning.

This repository accompanies the manuscript **“Multi-Agent Cooperative Defense via Role-Based Rewards and Dynamic Role Switching.”** The project investigates how multiple UAVs can protect an asset against an approaching threat UAV by coordinating **interceptor** and **escort** roles under limited LiDAR sensing.

> **Release status:** Source code and a summary of simulation results are included. Raw evaluation datasets, trained checkpoints, and standalone result files are not included. An independent clean-install reproduction has not been verified.

## Proposed approach

- **Role-based rewards:** An interceptor is rewarded for approaching and intercepting the threat; escorts are rewarded for maintaining protective positions near the asset.
- **Dynamic role switching:** A centralized assignment rule selects an interceptor among UAVs detecting the threat and can reassign this role as detection and relative positions change. UAV actions are generated individually using a shared learned policy.
- **Curriculum learning:** The interception success radius is progressively reduced from 15 m to 1.5 m as the policy meets performance criteria.
- **LiDAR-based partial observability:** Threat detection and role assignment depend on limited sensing ranges.

## Repository structure

| Path | Contents |
| --- | --- |
| `train_UAV_obstacle.py` | UAV training entry point, AirSim environment wrapper, training configuration, and runner integration. |
| `eval_UAV.py` | Evaluation entry point for a trained policy in AirSim; requires a valid local checkpoint and appropriate simulation setup. |
| `3D_animation.py` | Utilities for visualizing recorded UAV trajectories and engagements. |
| `intercept_geometry.json` | Configuration/data file related to interception geometry. |
| `onpolicy/config_UAV.py` | UAV experiment arguments and default training settings. |
| `onpolicy/config.py` | General on-policy algorithm configuration. |
| `onpolicy/envs/` | UAV training/evaluation environments and environment wrappers. |
| `onpolicy/algorithms/r_mappo/` | Recurrent MAPPO policy networks, actor–critic components, and PPO training logic. |
| `onpolicy/algorithms/mat/` | Multi-agent transformer components retained from the base framework. |
| `onpolicy/algorithms/utils/` | Neural-network and algorithm utilities. |
| `onpolicy/runner/shared/` | Shared-policy runners, including UAV training and evaluation runners. |
| `onpolicy/runner/separated/` | Separated-policy runners retained from the base framework. |
| `onpolicy/utils/` | Rollout buffers and shared training utilities. |
| `onpolicy/scripts/` | Original framework scripts for other benchmark environments; not the primary UAV entry point. |
| `environment.yaml`, `requirements.txt`, `setup.py` | Dependency and installation specifications. |
| `LICENSE` | Existing MIT license and original copyright notice. |

### UAV environment variants

| File | Intended use / distinction |
| --- | --- |
| `onpolicy/envs/make_env_role_curriculum.py` | Dynamic role-based cooperative defense environment, selected by the supplied training entry point. |
| `onpolicy/envs/make_env_static_role_division_2.py` | Static/fixed role allocation comparison environment. |
| `onpolicy/envs/make_env_PN_Lidar.py` | LiDAR-based PN-style baseline environment. |
| `onpolicy/envs/make_env_PN_LIDAR_curriculurm.py` | PN-style baseline variant with curriculum-based interception radius. |
| `onpolicy/envs/make_env_curriculurm_X.py` | Additional environment variant for comparisons/ablation; verify its configuration before associating it with a specific evaluation result. |
| `onpolicy/envs/eval_env.py`, `eval_env_2.py`, `eval_env_3.py` | Alternative UAV evaluation environment implementations. |

**Important:** Reward functions, termination conditions, role assignment, and interception radii differ across variants. Do not assume that all baseline environments use the proposed approach's reward settings. The *single-role strategy* and the repository's *PN-style baseline* should not automatically be treated as identical experiments without verifying the corresponding configuration.

## Running the code

1. Configure and start a compatible **AirSim** simulation with the UAV names and LiDAR sensors expected by the selected environment.
2. Install dependencies using `environment.yaml` or `requirements.txt`, as appropriate for the local Python/CUDA/AirSim installation.
3. Review `onpolicy/config_UAV.py` and `train_UAV_obstacle.py` to select the agent count, environment variant, rollout settings, and logging options.
4. Start training from the repository root:

   ```bash
   python train_UAV_obstacle.py
   ```

5. For evaluation, configure a valid **local** model checkpoint path in `eval_UAV.py`, select the matching environment, and run:

   ```bash
   python eval_UAV.py
   ```

**Reproducibility notes:** These are entry points in the supplied code, not a verified clean-install reproduction procedure. The training wrapper currently hard-codes `self.num_agents = 4`, while some configurations/environments default to three followers; make the counts consistent. `eval_UAV.py` contains machine-specific Windows checkpoint paths that must be replaced. Performance depends on the scenario, sensor range, training seeds, and evaluation configuration.

## Experimental results

The following results summarize cooperative defense performance across role-assignment strategies, LiDAR sensing ranges, curriculum learning configurations, and team sizes.

### Evaluation setup and metrics

The main comparison uses one protected asset, three UAVs, and one incoming threat UAV in AirSim. The threat moves toward the asset at 10 m/s; initial UAV positions are randomized. For the main strategy comparison at a 30 m LiDAR sensing range, each strategy was independently trained **10 times**, with each trained policy evaluated over **100 episodes**. Tabulated results are reported as **mean ± standard deviation** across independent runs.

- **Defense success rate (`R_success`):** Percentage of evaluation episodes in which the threat is intercepted before reaching the protected asset.
- **Friendly collision rate (`R_collision`):** Percentage of episodes with a collision between allied UAVs.
- **Protected-asset hit rate (`R_hit`):** Percentage of episodes in which the threat reaches or collides with the protected asset.

These metrics are calculated independently and are not necessarily mutually exclusive.

### 1. Comparison of role-assignment strategies (LiDAR: 30 m)

| Strategy | Defense success rate | Friendly collision rate | Protected-asset hit rate |
| --- | ---: | ---: | ---: |
| **Dynamic-role (proposed)** | **84.3 ± 1.2%** | **0.0 ± 0.0%** | **15.7 ± 1.2%** |
| Fixed-role | 71.2 ± 1.4% | 0.0 ± 0.0% | 28.8 ± 1.4% |
| Single-role (all interceptors) | 57.4 ± 1.6% | 1.4 ± 1.0% | 42.6 ± 1.6% |

At 30 m, dynamic-role assignment improved defense success by **13.1 percentage points** over fixed-role assignment and **26.9 percentage points** over single-role assignment. It also achieved the lowest protected-asset hit rate.

### 2. Effect of LiDAR sensing range

The table reports **defense success rate** across sensing conditions; each strategy was trained separately for the corresponding sensing range.

| LiDAR range | Dynamic-role (proposed) | Fixed-role | Single-role |
| --- | ---: | ---: | ---: |
| 50 m | **94.3 ± 1.2%** | 89.5 ± 1.1% | 85.1 ± 0.7% |
| 40 m | **93.0 ± 0.7%** | 78.0 ± 1.3% | 75.4 ± 3.1% |
| 30 m | **84.3 ± 1.2%** | 71.2 ± 1.4% | 57.4 ± 1.6% |
| 20 m | **68.6 ± 2.0%** | 40.0 ± 3.6% | 18.2 ± 0.8% |
| 10 m | **15.8 ± 2.0%** | 13.2 ± 1.3% | 12.0 ± 0.8% |

The dynamic-role strategy achieved the highest success rate at every evaluated range. Its advantage was particularly pronounced at **20 m** (a **28.6-percentage-point** gain over fixed-role and a **50.4-percentage-point** gain over single-role). At **10 m**, all strategies deteriorated substantially, suggesting that very late detection leaves insufficient time for interception. 

### 3. Effect of curriculum learning

Under a **30 m** LiDAR range, dynamic-role assignment was evaluated with and without curriculum learning. The curriculum-based policy gradually reduced the interception radius from **15 m** to **1.5 m**, while the non-curriculum baseline used **1.5 m** throughout training.

With curriculum learning, the policy reached stable performance earlier and attained a final defense success rate of **approximately 86%**, compared with **approximately 80%** without curriculum learning. These are approximate values from the training curves rather than mean ± standard deviation statistics.

### 4. Effect of the number of UAVs (LiDAR: 30 m)

| UAVs | Defense success rate | Friendly collision rate | Protected-asset hit rate |
| --- | ---: | ---: | ---: |
| 3 | 84.3 ± 1.2% | 0.0 ± 0.0% | 15.7 ± 1.2% |
| 4 | 89.2 ± 1.4% | 0.0 ± 0.0% | 10.8 ± 1.4% |
| **5** | **94.5 ± 1.2%** | **0.0 ± 0.0%** | **5.5 ± 1.2%** |
| 7 | 87.9 ± 2.4% | 5.0 ± 1.9% | 12.1 ± 2.4% |

The **five-UAV** configuration achieved the highest defense success rate among the evaluated team sizes. Increasing the team size further to seven reduced success and increased friendly collisions, indicating that additional UAVs do not always improve coordination. Each team size used a separately trained policy.

**Scope:** These results describe AirSim simulation performance under the specified configurations, not verified real-world UAV performance. Exact reproduction may require additional scenario settings, trained models, and evaluation data.

## Relationship to the original MAPPO implementation

This project builds on the publicly released **MAPPO/on-policy** implementation associated with:

Chao Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* ([paper](https://arxiv.org/abs/2103.01955); [original repository](https://github.com/marlbenchmark/on-policy)).

