# Third-party notices

BIRD is released under the MIT License (see `LICENSE`). It reimplements published
reward-design methods and interfaces with published benchmarks, and in doing so it
contains code, assets or text derived from the third-party works listed below. Each is
used under its own license; verbatim license texts are in `licenses/`.

Benchmarks and learners that BIRD only *imports* at run time (Meta-World, Gymnasium,
HumanoidBench, the Assistax Python package and its IPPO trainer, Stable-Baselines3,
MuJoCo, JAX, PyTorch) are not redistributed here and are governed by their own licenses.

## Code and assets included in this repository

| Component | Upstream | License | Where it appears in BIRD | Changes |
|---|---|---|---|---|
| FastTD3 | [younggyoseo/FastTD3](https://github.com/younggyoseo/FastTD3) @ `229ed59` | MIT (BAIR Commons composite; includes LeanRL, CleanRL and further notices) — `licenses/FastTD3-LICENSE.txt` | `bird/components/fasttd3.py` | Ported into BIRD's training interface (candidate reward substituted for the environment reward; checkpointing and evaluation hooks added). |
| SimbaV2 | [dojeon-ai/SimbaV2](https://github.com/dojeon-ai/SimbaV2) | Apache-2.0 — `licenses/SimbaV2-Apache-2.0.txt` | `bird/components/simba_v2.py`, SimbaV2 networks used by `fasttd3.py` | Translated from JAX to PyTorch and adapted to BIRD's training interface. |
| Assistax | [assistive-autonomy/assistax](https://github.com/assistive-autonomy/assistax) @ `a7d94f4e` | Apache-2.0 — `licenses/Assistax-Apache-2.0.txt` | `bird/envs/assets/assistax/` (scenes and meshes); reward terms transcribed in `bird/envs/assistax.py`; IPPO hyperparameters in `bird/components/_ippo_*.py` | Meshes decimated and converted from OBJ to STL; mesh references repointed; see `bird/envs/assets/assistax/PROVENANCE.json`. |
| MuJoCo Menagerie — Franka Emika Panda | [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (via Assistax) | Apache-2.0 — `licenses/mujoco_menagerie-Apache-2.0.txt` | Panda model and meshes in `bird/envs/assets/assistax/` | As for Assistax. |
| Assistive Gym | [Healthcare-Robotics/assistive-gym](https://github.com/Healthcare-Robotics/assistive-gym) (via Assistax) | MIT, Copyright (c) 2019 Healthcare Robotics Lab — `licenses/assistive-gym-MIT.txt` | Wheelchair and bed meshes in `bird/envs/assets/assistax/` | As for Assistax. |
| Meta-World | [Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld) | MIT — `licenses/Metaworld-MIT.txt` | Success checks and reward structure re-implemented in `bird/envs/metaworld.py` | Re-implemented against BIRD's adapter interface. |

## Method prompts and templates

To reproduce published methods faithfully, BIRD's method configurations and components
quote prompt text, templates and environment abstractions from the original papers and
their released code. Each quotation is attributed where it appears.

| Method | Source | License of the source |
|---|---|---|
| Eureka | [eureka-research/Eureka](https://github.com/eureka-research/Eureka) | MIT — `licenses/Eureka-MIT.txt` |
| DrEureka | [eureka-research/DrEureka](https://github.com/eureka-research/DrEureka) | MIT — `licenses/DrEureka-MIT.txt` |
| REvolve | [RishiHazra/Revolve](https://github.com/RishiHazra/Revolve) | MIT — `licenses/REvolve-MIT.txt` |
| LIMEN | [Lossfunk/LIMEN](https://github.com/Lossfunk/LIMEN) | MIT — `licenses/LIMEN-MIT.txt` |
| LaRes | [yeshenpy/LaRes](https://github.com/yeshenpy/LaRes) | MIT — `licenses/LaRes-MIT.txt` |
| Language to Rewards | [google-deepmind/language_to_reward_2023](https://github.com/google-deepmind/language_to_reward_2023) | Apache-2.0 — `licenses/language_to_reward-Apache-2.0.txt` |
| RF-Agent | [deng-ai-lab/RF-Agent](https://github.com/deng-ai-lab/RF-Agent); Gao et al., arXiv:2602.23876 | Paper CC BY 4.0 (most prompts are printed in its appendix); the code repository states no license |
| RDA | Lee et al., 2026 (paper) | CC BY 4.0 (paper) |
| Gran Turismo | Ma et al., 2025 (paper) | CC BY 4.0 (paper) |
| Text2Reward | [xlang-ai/text2reward](https://github.com/xlang-ai/text2reward); Xie et al., arXiv:2309.11489 | Paper CC BY 4.0; the code repository states no license |
| CARD | [ShengjieSun419/CARD](https://github.com/ShengjieSun419/CARD); Sun et al., arXiv:2410.14660 | The code repository states no license |
| R\* | Li et al., ICML 2025 (PMLR 267) | Published proceedings |

Short passages from works whose source states no license (Text2Reward, CARD, RF-Agent, R\*) are
quoted only to the extent needed to reproduce the published method, with attribution.
