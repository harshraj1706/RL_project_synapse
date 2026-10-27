# PROJECT BRIEF: SYNAPSE ARENA

## 1. Executive Summary
**SYNAPSE ARENA** is an autonomous reinforcement learning racing competition organized by **VECTOR × AIC** at **IIIT Bhagalpur, India**.

Teams are tasked with designing and training an autonomous AI racing agent that controls a vehicle in a simulated environment. The core challenge is **generalization**: agents are trained on official training maps but will be evaluated during the final races on **previously unseen tracks**.

- **Official Environment & Baseline Release:** Expected ~October 26–27, 2026.
- **Competition Final Sessions:** October 30 & November 1, 2026.
- **Key Constraints:** Shared training-step budget, limited training window, unseen evaluation tracks.

---

## 2. Core Engineering Principles
1. **Generalization Over Memorization:**
   Racing lines and speed profiles must generalize. Policies that memorize coordinate waypoints will fail when transferred to novel track geometries.
2. **Modular Architecture:**
   The training and evaluation infrastructure must be decoupled from the simulator interface via clear adapter boundaries.
3. **Evidence-Based Experimentation:**
   Every hyperparameter adjustment or reward shaping modification must be benchmarked against an established baseline using identical evaluation seeds and tracks.
4. **Step-Budget Accounting:**
   Training compute is constrained. Exploratory test runs must be tiny (smoke tests), reserving the step budget for verified, high-value training runs.

---

## 3. High-Level Architecture

```
                  +--------------------------------+
                  |  Official Racing Simulator     |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  |  OfficialEnvironmentAdapter    |
                  |  (Gymnasium Wrapper Boundary)  |
                  +---------------+----------------+
                                  |
                                  v
                  +--------------------------------+
                  |  Reward Shaper (Pass-through   |
                  |  or Optional Custom Shaping)   |
                  +---------------+----------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|  Training Pipeline    |                   |  Evaluation Pipeline  |
|  (PPO / Stable-       |                   |  (Deterministic       |
|   Baselines3)         |                   |   Policy Runner)      |
+-----------+-----------+                   +-----------+-----------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
| Checkpoints & Logs    |                   | CSV Metric Reports    |
| (weights, step logs)  |                   | (return, length, laps)|
+-----------------------+                   +-----------------------+
```

---

## 4. Multi-Phase Roadmap

### Phase 1: Foundation & Pipeline Plumbing (Current)
- Establish modular package structure and configuration validation.
- Implement abstract environment boundary and unconfigured official adapter guard.
- Build isolated synthetic smoke-test environment for rapid local integration testing.
- Implement minimal training and evaluation runners.
- Establish automated unit test suite.

### Phase 2: Official Simulator Inspection & Integration
- Inspect organizer-provided files, observation/action definitions, simulator API, and evaluator scripts.
- Document exact contracts in `docs/ENVIRONMENT_CONTRACT.md`.
- Implement `OfficialEnvironmentAdapter` to bridge the official simulator to Gymnasium.
- Run organizer-provided baseline without modifications to establish benchmark performance.

### Phase 3: Baseline Training & Controlled Iteration
- Train initial PPO baseline on official training tracks.
- Analyze driving behavior, stability, and failure modes (spinning, off-track, wall collisions).
- Iterate on observation normalization, network architecture, and conservative reward shaping.
- Benchmark generalization on held-out training maps or seeds.

### Phase 4: Model Selection & Submission Packaging
- Select the best-performing, most stable checkpoint across multiple seeds.
- Package the agent into the exact submission format required by the organizers.
- Run pre-submission verification using the official evaluation harness.
