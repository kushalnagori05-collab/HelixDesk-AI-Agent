"""
HelixDesk OpenEnv — Standard API Server + Gradio Dashboard.

Required for hackathon submission:
1. Returns 200 on root URL.
2. Exposes /reset, /step, /state endpoints for automated evaluation.
3. Provides a visual Gradio UI for manual review.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import gradio as gr
from pydantic import BaseModel
from typing import List, Optional, Any

from helixdesk import HelixDeskEnv
from helixdesk.agents import RuleAgent, RandomAgent

# Initialize global env for API
env = HelixDeskEnv()

# --- FastAPI Setup ---
app = FastAPI(title="HelixDesk OpenEnv API")

@app.get("/")
@app.get("/health")
async def health():
    return {"status": "ok", "env": "HelixDesk OpenEnv", "version": "1.0.0"}

@app.post("/reset")
async def reset():
    obs, info = env.reset()
    # Convert numpy to list for JSON
    return {
        "observation": obs.tolist(),
        "info": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in info.items()}
    }

class StepRequest(BaseModel):
    action: List[int]

@app.post("/step")
async def step(req: StepRequest):
    obs, reward, terminated, truncated, info = env.step(req.action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in info.items()}
    }

@app.get("/state")
async def get_state():
    obs = env.state()
    return {"observation": obs.tolist()}

# --- Gradio UI Logic ---
def run_episode(agent_type: str) -> tuple:
    """Run a full episode and return chart figures + summary markdown."""
    local_env = HelixDeskEnv()
    if agent_type == "rule":
        agent = RuleAgent(local_env.observation_space, local_env.action_space)
    else:
        agent = RandomAgent(local_env.observation_space, local_env.action_space)

    obs, info = local_env.reset(seed=42)
    steps, cumulative_rewards = [], []
    queue_depths, overdue_counts, csat_scores = [], [], []
    ep_reward = 0.0
    done = False
    while not done:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = local_env.step(action)
        ep_reward += reward
        done = terminated or truncated
        steps.append(info.get("step", 0))
        cumulative_rewards.append(ep_reward)
        queue_depths.append(info.get("queue_depth", 0))
        overdue_counts.append(info.get("overdue_count", 0))
        if info.get("csat_score") is not None:
            csat_scores.append(float(info["csat_score"]))

    # Charts
    fig_reward, ax_reward = plt.subplots(figsize=(7, 3.5))
    ax_reward.plot(steps, cumulative_rewards, color="#4f46e5", linewidth=2)
    ax_reward.set_title("Reward", fontweight="bold")
    fig_reward.tight_layout()

    fig_queue, ax_queue = plt.subplots(figsize=(7, 3.5))
    ax_queue.plot(steps, queue_depths, color="#0891b2", linewidth=2)
    ax_queue.set_title("Queue Depth", fontweight="bold")
    fig_queue.tight_layout()

    summary = f"### Episode Summary\n| Metric | Value |\n|---|---|\n| **Agent** | {agent_type} |\n| **Reward** | {ep_reward:.2f} |"
    return fig_reward, fig_queue, summary

# Create Gradio interface
with gr.Blocks(title="HelixDesk OpenEnv", theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 📧 HelixDesk OpenEnv\nGymnasium RL for customer email queue management.")
    with gr.Row():
        agent_dropdown = gr.Dropdown(choices=["rule", "random"], value="rule", label="Agent")
        run_btn = gr.Button("▶ Run Episode", variant="primary")
    with gr.Row():
        reward_plot, queue_plot = gr.Plot(label="Reward"), gr.Plot(label="Queue")
    summary_md = gr.Markdown("Click run to see results")
    run_btn.click(run_episode, [agent_dropdown], [reward_plot, queue_plot, summary_md])

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
