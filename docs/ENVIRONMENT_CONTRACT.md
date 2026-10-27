# ENVIRONMENT CONTRACT & INTEGRATION GUIDE

This document defines the boundary between the **SYNAPSE ARENA** training pipeline and the official racing simulator.

---

## 1. Current Status: Awaiting Official Environment
As of October 26, 2026, the official environment package, training maps, baseline agent, and submission specifications have not yet been provided by the competition organizers.

To avoid assumptions and bugs:
- **No official API signatures or observation fields are assumed.**
- The `OfficialEnvironmentAdapter` (`src/synapse_arena/envs/official_adapter.py`) intentionally raises a descriptive `EnvironmentNotIntegratedError` if instantiated before official integration.
- A clearly labeled `SyntheticSmokeTestRacingEnv` is provided strictly to verify software plumbing. It must never be used to represent racing performance.

---

## 2. Environment Specification Matrix

| Aspect | Confirmed Facts | Unknowns (Pending Discovery) | Discovery Strategy |
| :--- | :--- | :--- | :--- |
| **Package / Engine** | Simulated car racing; autonomous AI control | Custom Pygame, Unity, Box2D, Carla, AirSim, or custom Python C-extension | Inspect official repository `setup.py`, `pyproject.toml`, or README |
| **Observation Space** | Not yet confirmed | Vector (LIDAR, raycasts, speed, angle) vs Image (RGB/grayscale) vs Dict | Print `env.observation_space`, inspect simulator `step()` return |
| **Action Space** | Not yet confirmed | Continuous (steering, throttle, brake) vs Discrete (left, right, gas, brake) | Print `env.action_space`, inspect car dynamics code |
| **Reset & Step API** | Expected to follow standard step loop | Gymnasium `(obs, reward, terminated, truncated, info)` vs classic Gym `(obs, reward, done, info)` | Inspect simulator class methods and return tuple length |
| **Track Selection** | Training tracks provided; finals use unseen tracks | Track format, map loading parameter (e.g. `track_name="oval"`), seed parameter | Inspect environment `__init__` or `reset(options=...)` |
| **Official Reward** | Progress / speed / survival reward | Exact formula, penalties for collision, checkpoint rewards | Inspect official reward calculation function |
| **Evaluator Protocol**| Official evaluation runs on unseen tracks | CLI runner, docker container, or callable function `evaluate(agent)` | Inspect organizer baseline and evaluation scripts |
| **Step Accounting**  | Strict shared step budget | Server-side tracking vs local step counter | Review competition rulebook and evaluator logs |

---

## 3. Official Simulator Integration Checklist (Phase 2)

When the official files arrive, complete the following inspection checklist before modifying training code:

1. [ ] **Inspect Documentation & Setup:**
   - Read the official README, installation guide, and dependency requirements.
2. [ ] **Inspect Observation Space:**
   - Record exact dimensionality, data types (`float32`, `uint8`), bounds (`low`, `high`), and semantic meaning of each feature.
3. [ ] **Inspect Action Space:**
   - Record control bounds, discrete vs. continuous, and control scaling (e.g., steering range `[-1.0, 1.0]`).
4. [ ] **Inspect Reward & Termination Logic:**
   - Verify what constitutes `terminated` (crash, out of bounds, lap complete) vs `truncated` (time limit reached).
   - Document the base reward components.
5. [ ] **Inspect Multi-Track & Seeding Support:**
   - Check how maps and random seeds are passed to the environment.
6. [ ] **Integrate `OfficialEnvironmentAdapter`:**
   - Implement the adapter in `src/synapse_arena/envs/official_adapter.py` wrapping the official simulator into a standard `gymnasium.Env`.
7. [ ] **Verify with Tests:**
   - Run `gymnasium.utils.env_checker.check_env` on the integrated adapter.
   - Run `tests/test_env_boundary.py` to confirm compatibility.
