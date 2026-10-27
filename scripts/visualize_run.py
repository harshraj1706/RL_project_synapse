#!/usr/bin/env python
"""Live Telemetry and Visualizer Runner for SYNAPSE ARENA racing policies.

Runs a trained policy or baseline agent through an episode and prints real-time
telemetry to the console, plus exports an interactive HTML5 visualizer.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from stable_baselines3 import PPO
from synapse_arena.config.schema import load_config
from synapse_arena.envs.factory import make_racing_env
from synapse_arena.rewards.shaping import RewardShaper
from synapse_arena.utils.logger import get_logger

logger = get_logger("synapse_arena.visualize")


def make_gauge(value: float, min_val: float, max_val: float, width: int = 16) -> str:
    """Renders a simple ASCII gauge bar."""
    clamped = max(min_val, min(max_val, value))
    fraction = (clamped - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    filled = int(round(fraction * width))
    empty = width - filled
    return f"[{'#' * filled}{'-' * empty}]"


def make_steer_gauge(value: float, width: int = 15) -> str:
    """Renders a bi-directional steering gauge centered at 0."""
    half = width // 2
    clamped = max(-1.0, min(1.0, value))
    pos = int(round((clamped + 1.0) / 2.0 * (width - 1)))
    chars = ["-"] * width
    chars[half] = "|"
    chars[pos] = "<" if pos < half else (">" if pos > half else "^")
    return f"[{''.join(chars)}]"


def generate_html_viewer(frames: list, output_path: Path) -> None:
    """Generates a self-contained HTML5 2D visualizer to watch the car's trajectory."""
    data_json = json.dumps(frames)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SYNAPSE ARENA — Telemetry Visualizer</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0d1117;
      color: #c9d1d9;
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }}
    .header {{
      text-align: center;
      margin-bottom: 16px;
    }}
    .header h1 {{
      color: #58a6ff;
      margin: 0 0 6px 0;
      font-size: 24px;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 10px;
      border-radius: 12px;
      font-size: 12px;
      background: #238636;
      color: #ffffff;
      font-weight: bold;
    }}
    .card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 16px;
      margin-bottom: 16px;
      width: 860px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }}
    .canvas-container {{
      display: flex;
      justify-content: center;
      background: #090d13;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px;
    }}
    canvas {{
      background: #090d13;
      display: block;
    }}
    .controls {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 14px;
      gap: 12px;
    }}
    button {{
      background: #238636;
      border: 1px solid #2ea043;
      color: #ffffff;
      font-weight: bold;
      padding: 8px 18px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
    }}
    button:hover {{
      background: #2ea043;
    }}
    .slider {{
      flex-grow: 1;
      cursor: pointer;
    }}
    .dashboard {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
      margin-top: 14px;
    }}
    .stat-box {{
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 6px;
      padding: 10px;
      text-align: center;
    }}
    .stat-label {{
      font-size: 11px;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .stat-val {{
      font-size: 18px;
      font-weight: bold;
      color: #58a6ff;
      margin-top: 4px;
    }}
    .sensors-grid {{
      display: flex;
      justify-content: space-around;
      margin-top: 10px;
      padding: 8px;
      background: #0d1117;
      border-radius: 6px;
      border: 1px solid #30363d;
      font-family: monospace;
      font-size: 13px;
    }}
    .sensor-item {{
      text-align: center;
    }}
    .sensor-val {{
      color: #7ee787;
      font-weight: bold;
    }}
  </style>
</head>
<body>
  <div class="header">
    <h1>SYNAPSE ARENA — Telemetry & Behavior Visualizer</h1>
    <span class="badge">Pipeline Verification Mode (Synthetic Test Environment)</span>
  </div>

  <div class="card">
    <div class="canvas-container">
      <canvas id="viewCanvas" width="820" height="420"></canvas>
    </div>

    <div class="controls">
      <button id="playBtn" onclick="togglePlay()">Pause</button>
      <button onclick="resetPlayback()">Restart</button>
      <input type="range" id="frameSlider" class="slider" min="0" max="{len(frames)-1}" value="0" oninput="onSlide(this.value)">
      <span id="stepLabel" style="font-family: monospace; font-size: 14px;">Step: 0 / {len(frames)-1}</span>
    </div>

    <div class="dashboard">
      <div class="stat-box">
        <div class="stat-label">Forward Speed</div>
        <div class="stat-val" id="statSpeed">0.00</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Steering Output</div>
        <div class="stat-val" id="statSteer">0.00</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Throttle / Brake</div>
        <div class="stat-val" id="statThrottle">0.00</div>
      </div>
      <div class="stat-box">
        <div class="stat-label">Cumulative Return</div>
        <div class="stat-val" id="statReturn" style="color: #f2cc60;">0.00</div>
      </div>
    </div>

    <div class="sensors-grid">
      <div class="sensor-item"><div>Left 90°</div><div class="sensor-val" id="s0">-</div></div>
      <div class="sensor-item"><div>Left 45°</div><div class="sensor-val" id="s1">-</div></div>
      <div class="sensor-item"><div>Center 0°</div><div class="sensor-val" id="s2">-</div></div>
      <div class="sensor-item"><div>Right 45°</div><div class="sensor-val" id="s3">-</div></div>
      <div class="sensor-item"><div>Right 90°</div><div class="sensor-val" id="s4">-</div></div>
    </div>
  </div>

  <script>
    const frames = {data_json};
    let currentIdx = 0;
    let isPlaying = true;
    let timer = null;

    const canvas = document.getElementById("viewCanvas");
    const ctx = canvas.getContext("2d");

    function draw() {{
      const f = frames[currentIdx];
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Grid
      ctx.strokeStyle = "#1b212c";
      ctx.lineWidth = 1;
      for (let x = 0; x < canvas.width; x += 40) {{
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      }}
      for (let y = 0; y < canvas.height; y += 40) {{
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
      }}

      // Calculate origin & scaling
      const originX = 60;
      const originY = canvas.height / 2;
      const scale = 7.0;

      // Draw path history
      ctx.beginPath();
      ctx.strokeStyle = "rgba(88, 166, 255, 0.4)";
      ctx.lineWidth = 3;
      for (let i = 0; i <= currentIdx; i++) {{
        const px = originX + frames[i].x * scale;
        const py = originY + frames[i].y * scale;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }}
      ctx.stroke();

      // Current car position
      const carX = originX + f.x * scale;
      const carY = originY + f.y * scale;
      const heading = f.heading;

      // Draw 5 sensor beams
      const beamAngles = [-Math.PI/2, -Math.PI/4, 0, Math.PI/4, Math.PI/2];
      const sensorDists = [f.s0, f.s1, f.s2, f.s3, f.s4];

      for (let i = 0; i < 5; i++) {{
        const bAngle = heading + beamAngles[i];
        const bLen = (sensorDists[i] || 25.0) * 1.5;
        const endX = carX + Math.cos(bAngle) * bLen;
        const endY = carY + Math.sin(bAngle) * bLen;

        ctx.beginPath();
        ctx.strokeStyle = (i === 2) ? "rgba(255, 123, 114, 0.7)" : "rgba(126, 231, 135, 0.5)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        ctx.moveTo(carX, carY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      // Draw Car Body (rotated rectangle)
      ctx.save();
      ctx.translate(carX, carY);
      ctx.rotate(heading);

      // Chassis
      ctx.fillStyle = "#58a6ff";
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.roundRect(-14, -8, 28, 16, 3);
      ctx.fill();
      ctx.stroke();

      // Windshield
      ctx.fillStyle = "#0d1117";
      ctx.fillRect(0, -6, 6, 12);

      // Headlights
      ctx.fillStyle = "#ffe066";
      ctx.fillRect(12, -7, 3, 4);
      ctx.fillRect(12, 3, 3, 4);

      ctx.restore();

      // Update dashboard values
      document.getElementById("statSpeed").innerText = f.speed.toFixed(3);
      document.getElementById("statSteer").innerText = f.steering.toFixed(3);
      document.getElementById("statThrottle").innerText = f.throttle.toFixed(3);
      document.getElementById("statReturn").innerText = f.cum_reward.toFixed(2);
      document.getElementById("stepLabel").innerText = `Step: ${{f.step}} / ${{frames.length - 1}}`;
      document.getElementById("frameSlider").value = currentIdx;

      document.getElementById("s0").innerText = f.s0.toFixed(1) + "m";
      document.getElementById("s1").innerText = f.s1.toFixed(1) + "m";
      document.getElementById("s2").innerText = f.s2.toFixed(1) + "m";
      document.getElementById("s3").innerText = f.s3.toFixed(1) + "m";
      document.getElementById("s4").innerText = f.s4.toFixed(1) + "m";
    }}

    function tick() {{
      if (isPlaying) {{
        currentIdx = (currentIdx + 1) % frames.length;
        draw();
      }}
    }}

    function togglePlay() {{
      isPlaying = !isPlaying;
      document.getElementById("playBtn").innerText = isPlaying ? "Pause" : "Play";
    }}

    function resetPlayback() {{
      currentIdx = 0;
      draw();
    }}

    function onSlide(val) {{
      currentIdx = parseInt(val);
      draw();
    }}

    draw();
    timer = setInterval(tick, 100);
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_content, encoding="utf-8")


def run_visual_episode(config_path: str = "configs/default.yaml", checkpoint_path: str = "checkpoints/smoke_test/model.zip"):
    config = load_config(config_path)
    ckpt = Path(checkpoint_path)

    logger.info("Loading model from %s", ckpt)
    model = PPO.load(ckpt, device=config.training.device)

    raw_env = make_racing_env(config.env, seed=config.seed)
    env = RewardShaper(raw_env, config.reward)

    obs, info = env.reset()
    done = False
    cum_reward = 0.0
    step = 0

    frames = []

    print("\n" + "=" * 90)
    print("  SYNAPSE ARENA - AGENT TELEMETRY RUNNER (Synthetic Test Environment)")
    print("=" * 90)
    print(f"{'Step':^6} | {'Steering':^17} | {'Throttle':^14} | {'Speed':^16} | {'Step Rew':^9} | {'Cum Return':^10}")
    print("-" * 90)

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        step += 1
        done = bool(terminated or truncated)

        steering = float(action[0])
        throttle = float(action[1])
        speed = float(info.get("speed", 0.0))
        x = float(info.get("x", 0.0))
        y = float(info.get("y", 0.0))
        heading = float(info.get("heading", 0.0))

        # Sensor readings from obs
        s0, s1, s2, s3, s4 = [float(obs[i]) for i in range(5)]

        frames.append({
            "step": step,
            "steering": steering,
            "throttle": throttle,
            "speed": speed,
            "x": x,
            "y": y,
            "heading": heading,
            "step_reward": reward,
            "cum_reward": cum_reward,
            "s0": s0,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
        })

        # Print console telemetry every 5 steps, or first 5 and last 5
        if step <= 5 or step % 10 == 0 or done:
            steer_bar = make_steer_gauge(steering, 13)
            throttle_bar = make_gauge(throttle, -1.0, 1.0, 10)
            speed_bar = make_gauge(speed, 0.0, 1.0, 10)
            print(f"{step:^6} | {steering:+.2f} {steer_bar} | {throttle:+.2f} {throttle_bar} | {speed:.2f} {speed_bar} | {reward:^9.2f} | {cum_reward:^10.2f}")

    print("=" * 90)
    print(f"Run Finished! Total Steps: {step} | Total Return: {cum_reward:.2f}")

    html_path = Path("eval_results/trajectory_viewer.html")
    generate_html_viewer(frames, html_path)
    print(f"\n[Artifact Created] Interactive 2D Visualizer saved to: {html_path.resolve()}")
    print("Open this HTML file in your browser to watch the car move, turn, and view live sensor beams!\n")

    env.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize a trained racing agent run.")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/smoke_test/model.zip")
    args = parser.parse_args()

    run_visual_episode(args.config, args.checkpoint)


if __name__ == "__main__":
    main()
