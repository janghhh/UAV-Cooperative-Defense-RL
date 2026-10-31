# Cooperative Multi-UAV Defense Using Multi-Agent Reinforcement Learning

Research code for AirSim-based cooperative UAV defense experiments using multi-agent proximal policy optimization (MAPPO/rMAPPO), with role-based and comparison environments.

> **Research release status:** Source code is included. The experimental datasets, evaluation results, trained checkpoints, and paper citation will be added separately when they are ready for public release. No experimental results are claimed in this README.

## Repository structure

| Path | Contents |
| --- | --- |
| `train_UAV_obstacle.py` | UAV training entry point, AirSim environment wrapper, training configuration, and runner integration. |
| `eval_UAV.py` | Evaluation entry point for a trained policy in AirSim. Requires a valid checkpoint path and appropriate AirSim configuration. |
| `3D_animation.py` | Utilities for visualizing recorded UAV trajectories and engagements. |
| `intercept_geometry.json` | Configuration/data file related to interception geometry. |
| `onpolicy/config_UAV.py` | UAV experiment arguments and default training settings. |
| `onpolicy/config.py` | General on-policy algorithm configuration. |
| `onpolicy/envs/` | UAV training and evaluation environments, plus vectorized environment wrappers. See variant table below. |
| `onpolicy/algorithms/r_mappo/` | Recurrent MAPPO policy networks, actor–critic components, and PPO training logic. |
| `onpolicy/algorithms/mat/` | Multi-agent transformer algorithm components retained from the base framework. |
| `onpolicy/algorithms/utils/` | Shared neural-network and algorithm utilities. |
| `onpolicy/runner/shared/` | Shared-policy runners, including UAV rollout/training/evaluation runners. |
| `onpolicy/runner/separated/` | Separated-policy runners retained from the base framework. |
| `onpolicy/utils/` | Replay buffers and shared training utilities. |
| `onpolicy/scripts/` | Original framework scripts for other benchmark environments; not the primary UAV experiment entry point. |
| `environment.yaml`, `requirements.txt`, `setup.py` | Dependency and installation specifications. |
| `LICENSE` | Existing MIT license and original copyright notice. |

### UAV environment variants

| File | Intended use / distinction |
| --- | --- |
| `onpolicy/envs/make_env_role_curriculum.py` | Dynamic role-based cooperative defense environment; currently selected by the training entry point. |
| `onpolicy/envs/make_env_static_role_division_2.py` | Static/fixed role allocation comparison environment. |
| `onpolicy/envs/make_env_PN_Lidar.py` | LiDAR-based PN-style baseline environment. |
| `onpolicy/envs/make_env_PN_LIDAR_curriculurm.py` | PN-style baseline variant with a curriculum-based interception radius. |
| `onpolicy/envs/make_env_curriculurm_X.py` | Additional environment variant used for comparisons/ablation; inspect configuration before treating it as a specific published experiment. |
| `onpolicy/envs/eval_env.py`, `eval_env_2.py`, `eval_env_3.py` | Alternative UAV evaluation environment implementations. |

**Important:** Reward functions, termination conditions, role-assignment mechanisms, and interception radii may differ between variants. Baseline environments should not be assumed to use the proposed method's reward settings.

## Running the code

1. Configure and start a compatible **AirSim** simulation with the UAV names and LiDAR sensors expected by the selected environment.
2. Install dependencies from `environment.yaml` or `requirements.txt` as appropriate for the local Python/CUDA/AirSim setup.
3. Review `onpolicy/config_UAV.py` and `train_UAV_obstacle.py` to set the agent count, environment variant, rollout settings, and logging options.
4. Start training from the repository root with `python train_UAV_obstacle.py`.
5. For evaluation, set a valid **local** model checkpoint path in `eval_UAV.py`, choose the corresponding evaluation environment, and run `python eval_UAV.py`.

These commands describe the entry points in the supplied code; a clean-install, end-to-end reproduction has **not** been independently verified. In particular, `train_UAV_obstacle.py` currently hard-codes `self.num_agents = 4` in its wrapper, while some configuration/environment defaults use three followers. **Make agent counts consistent before running**. `eval_UAV.py` also contains machine-specific Windows checkpoint paths; replace them with paths relevant to your setup. Training requires AirSim and the correct vehicle/sensor definitions.

## Planned experimental materials (not yet included)

The following directories are reserved for later releases; they are **not** included in this initial package:

| Planned path | Planned contents |
| --- | --- |
| `data/` | Experiment inputs, scenario settings, and non-sensitive evaluation data. |
| `results/` | Quantitative results, comparative tables, and figures from the paper. |
| `checkpoints/` | Trained model weights, if they can be shared. |
| `docs/` | Additional experimental protocol, reproducibility notes, and publication details. |

When results are uploaded, include the exact experiment configuration, seeds, number of trials, definitions of metrics, and the code variant used to produce each result.

## Relationship to the original MAPPO implementation

This project builds on the publicly released MAPPO/on-policy implementation associated with:

Chao Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* ([paper](https://arxiv.org/abs/2103.01955); [original repository](https://github.com/marlbenchmark/on-policy)).

The repository retains the existing MIT `LICENSE` and original contributor attribution. UAV experiment environments and scripts are project-specific adaptations. Please cite the underlying MAPPO work when using its implementation, alongside the associated UAV study when citation information is published.

## Citation

**The UAV paper's citation information will be added here.**

## License

See [LICENSE](LICENSE). Respect the licenses and citation requirements of external software, including AirSim and any dependencies.
