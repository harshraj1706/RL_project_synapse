# AGENTS.md — Persistent Instructions for AI Agents

This document defines the persistent engineering standards, operational rules, and development guidelines for all AI agents working on **SYNAPSE ARENA** (VECTOR × AIC, IIIT Bhagalpur).

## 1. Prime Directive: Truthfulness and Verification
- **Never fabricate simulator specifications**: Do not invent import paths, action bounds, observation channels, coordinate systems, or reward terms.
- **Distinguish confirmed facts from hypotheses**: If an environment property is unverified, treat it as unknown and document it in `docs/ENVIRONMENT_CONTRACT.md`.
- **Never claim unverified results**: A smoke test on a synthetic environment tests *software integration*, not *racing capability*. Never describe synthetic test runs as racing performance or lap completion.
- **Always verify changes with automated tests**: Run `pytest` and relevant smoke tests before declaring any task complete.

## 2. Competition & Step-Budget Discipline
- **Respect the training step budget**: During Phase 1 and local testing, only run tiny smoke tests (e.g. 64-100 steps). Do not launch long training runs or consume training resources until the official environment is integrated and a deliberate experimental plan is agreed upon.
- **Never cheat or exploit evaluator flaws**: Do not hardcode specific track coordinates, use hidden simulator state not exposed to the agent, or bypass the official evaluator protocol. Generalization to unseen tracks is the primary success criterion.
- **Preserve baselines and checkpoints**: Never overwrite official baselines or previous best checkpoints without explicit confirmation.

## 3. Architecture & Code Quality Guidelines
- **Maintain the environment boundary**: All interaction with the racing simulator must pass through `synapse_arena.envs.base.BaseRacingEnv` and its adapters. The training loop and agent logic must remain decoupled from specific simulator quirks.
- **Keep reward shaping optional and isolated**: The raw environment reward must be passed through unchanged by default. Any custom reward shaping must be implemented via `synapse_arena.rewards.shaping.RewardShaper` and explicitly enabled in the configuration file.
- **Strict configuration management**: All parameters (seeds, hyperparameters, environment IDs, paths) belong in YAML configuration files validated through `synapse_arena.config.schema.ConfigSchema`. No hardcoded magic constants in code.
- **Deterministic and reproducible**: Explicitly set random seeds across Python `random`, `numpy`, PyTorch, and Gymnasium environments using `synapse_arena.utils.seed.set_seed()`.
- **Windows compatibility**: Always use `pathlib.Path` for cross-platform file paths. Keep PowerShell execution syntax in mind.

## 4. Git & File Operations
- **Do not automatically push, commit, or delete**: Never delete user checkpoints or push to remote Git repositories without explicit user instruction.
- **Preserve user modifications**: Inspect existing files and Git status before making changes.

## 5. Testing Protocol
Any modifications to the codebase must pass:
1. `pytest tests/` — Unit tests for configs, reward shaping, and environment boundaries.
2. `python scripts/train.py --config configs/default.yaml` — End-to-end smoke test verifying pipeline integrity.
3. `python scripts/evaluate.py --config configs/default.yaml --checkpoint checkpoints/smoke_test/model.zip` — Deterministic evaluation verification.
