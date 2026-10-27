"""Training orchestration pipeline for SYNAPSE ARENA."""

from pathlib import Path
from stable_baselines3 import PPO
from synapse_arena.config.schema import AppConfig, save_config
from synapse_arena.envs.factory import make_racing_env
from synapse_arena.rewards.shaping import RewardShaper
from synapse_arena.training.callbacks import ProgressLoggerCallback
from synapse_arena.utils.logger import get_logger
from synapse_arena.utils.seed import set_seed

logger = get_logger(__name__)


def train_policy(config: AppConfig) -> Path:
    """Executes policy training using the provided configuration.

    Args:
        config: Application configuration.

    Returns:
        Path: Path to the saved trained model (.zip).
    """
    logger.info("Initializing training session with seed=%d...", config.seed)
    set_seed(config.seed)

    # 1. Environment creation & reward wrapping
    logger.info("Creating environment: name='%s', track='%s'", config.env.name, config.env.track)
    raw_env = make_racing_env(config.env, seed=config.seed)
    env = RewardShaper(raw_env, config.reward)

    # 2. Setup output directories
    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config.training.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Model initialization (or resume from checkpoint)
    if config.training.resume_checkpoint:
        resume_path = Path(config.training.resume_checkpoint)
        logger.info("Resuming training from checkpoint: %s", resume_path)
        model = PPO.load(resume_path, env=env, device=config.training.device)
    else:
        logger.info(
            "Instantiating new %s agent (policy=%s, lr=%.5f, n_steps=%d, batch_size=%d)",
            config.algorithm.name,
            config.algorithm.policy,
            config.algorithm.learning_rate,
            config.algorithm.n_steps,
            config.algorithm.batch_size,
        )
        model = PPO(
            policy=config.algorithm.policy,
            env=env,
            learning_rate=config.algorithm.learning_rate,
            n_steps=config.algorithm.n_steps,
            batch_size=config.algorithm.batch_size,
            gamma=config.algorithm.gamma,
            gae_lambda=config.algorithm.gae_lambda,
            clip_range=config.algorithm.clip_range,
            ent_coef=config.algorithm.ent_coef,
            verbose=1,
            seed=config.seed,
            device=config.training.device,
        )

    # 4. Execute training
    logger.info("Starting training run for %d total timesteps...", config.training.total_timesteps)
    progress_callback = ProgressLoggerCallback(log_freq=max(1, config.training.total_timesteps // 4))

    model.learn(
        total_timesteps=config.training.total_timesteps,
        callback=progress_callback,
        reset_num_timesteps=False if config.training.resume_checkpoint else True,
    )

    # 5. Save model checkpoint and active configuration
    saved_model_path = checkpoint_dir / "model.zip"
    model.save(saved_model_path)
    logger.info("Model checkpoint successfully saved to: %s", saved_model_path.resolve())

    saved_config_path = checkpoint_dir / "config.yaml"
    save_config(config, saved_config_path)
    logger.info("Experiment configuration saved to: %s", saved_config_path.resolve())

    env.close()
    return saved_model_path
