"""Generate a multi-policy UAVSim4Security defense comparison animation.

This script runs the same attack scenario with several routing policies and
writes a self-contained HTML dashboard. The default comparison is:

* greedy: simulator-sorted next-hop baseline
* random: legal random action baseline
* defensive: a hand-built proxy for the proposed PPO policy behavior

The defensive policy is not a trained PPO checkpoint. It is a visual proxy that
helps inspect whether the RL bridge exposes the right cross-layer signals before
running longer RLlib training.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import math
import random
from pathlib import Path
from typing import Optional

from uavsim4security_core_env import UAVSim4SecurityCoreEnv, UAVSim4SecurityEnvConfig


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "outputs" / "uavsim4security_cross_layer_demo.html"
POLICY_TITLES = {
    "greedy": "Greedy baseline",
    "random": "Random baseline",
    "defensive": "Proposed defense",
}
POLICY_DESCRIPTIONS = {
    "greedy": "Uses the simulator candidate ordering and always selects the first valid next hop.",
    "random": "Samples a legal masked action at every control step.",
    "defensive": "At each packet forwarding decision, avoids known malicious relays and active jammer coverage when alternatives exist.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--episode-duration", type=float, default=20.0)
    parser.add_argument("--control-interval", type=float, default=0.5)
    parser.add_argument("--num-uavs", type=int, default=6)
    parser.add_argument("--candidate-limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--mode", choices=("comparison", "single"), default="comparison")
    parser.add_argument("--policy", choices=("defensive", "greedy", "random"), default="defensive")
    parser.add_argument("--compare-policies", default="greedy,random,defensive")
    parser.add_argument("--attacks", default="BLACKHOLE,GRAYHOLE,PHY_JAMMING")
    parser.add_argument("--attacker-ids", default="4,5")
    parser.add_argument("--attack-probability", type=float, default=0.85)
    parser.add_argument("--sensing-range", type=float, default=190.0)
    parser.add_argument("--jammer-radius", type=float, default=170.0)
    parser.add_argument("--jammer-power-w", type=float, default=0.18)
    parser.add_argument("--jammer-coords", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attacks = _parse_names(args.attacks)
    attacker_ids = tuple(item for item in _parse_ints(args.attacker_ids) if 0 <= item < args.num_uavs)
    jammer_coords = _parse_coords(args.jammer_coords)
    base_config = UAVSim4SecurityEnvConfig(
        num_uavs=args.num_uavs,
        episode_duration=args.episode_duration,
        control_interval=args.control_interval,
        candidate_limit=args.candidate_limit,
        seed=args.seed,
        attacks=attacks,
        attacker_ids=attacker_ids,
        attack_probability=args.attack_probability,
        sensing_range=args.sensing_range,
        jammer_coords=jammer_coords,
        jammer_radius=args.jammer_radius,
        jammer_power_w=args.jammer_power_w,
    )

    policies = [args.policy] if args.mode == "single" else list(_parse_names(args.compare_policies.lower()))
    policies = [policy for policy in policies if policy in POLICY_TITLES]
    if not policies:
        raise ValueError("No valid policies selected.")

    runs = []
    for policy in policies:
        print(f"Running policy: {policy}")
        config = copy.copy(base_config)
        if policy == "defensive":
            config.route_policy = "defensive"
        frames = run_episode(config, policy, args.seed, args.max_steps, args.quiet)
        summary = summarize_frames(frames)
        runs.append(
            {
                "policy": policy,
                "title": POLICY_TITLES[policy],
                "description": POLICY_DESCRIPTIONS[policy],
                "frames": frames,
                "summary": summary,
            }
        )
        print(_format_summary(policy, summary))

    html = render_html(runs, base_config, args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote comparison animation: {args.output}")


def run_episode(
    config: UAVSim4SecurityEnvConfig,
    policy: str,
    seed: int,
    max_steps: int,
    quiet: bool,
) -> list[dict]:
    core = UAVSim4SecurityCoreEnv(config)
    rng = random.Random(seed + 91)
    frames: list[dict] = []

    stream = io.StringIO()
    redirect = contextlib.redirect_stdout(stream) if quiet else contextlib.nullcontext()
    with redirect:
        core.reset(seed=seed)
        first_frame = core.visualization_snapshot()
        first_frame["actions"] = {agent: 0 for agent in core.agents}
        first_frame["policy"] = policy
        frames.append(first_frame)
        terminated = False
        steps = 0
        while not terminated:
            actions = select_actions(core, policy, rng)
            result = core.step(actions)
            frame = core.visualization_snapshot()
            frame["actions"] = actions
            frame["policy"] = policy
            frames.append(frame)
            terminated = result.terminated or result.truncated
            steps += 1
            if max_steps and steps >= max_steps:
                break
    core.close()
    return frames


def select_actions(core: UAVSim4SecurityCoreEnv, policy: str, rng: random.Random) -> dict[str, int]:
    if policy == "greedy":
        return {agent: 0 for agent in core.agents}
    if policy == "random":
        return {agent: _random_legal_action(core, agent, rng) for agent in core.agents}
    return {agent: _defensive_action(core, agent) for agent in core.agents}


def _random_legal_action(core: UAVSim4SecurityCoreEnv, agent: str, rng: random.Random) -> int:
    mask = core.action_mask(agent)
    legal = [idx for idx, value in enumerate(mask) if value > 0.5]
    return rng.choice(legal) if legal else 0


def _defensive_action(core: UAVSim4SecurityCoreEnv, agent: str) -> int:
    candidates = core.candidate_next_hops(agent)
    if not candidates:
        return 0

    snapshot = core.visualization_snapshot()
    attackers = set(snapshot.get("attacker_ids", []))
    jammer = snapshot.get("jammer")
    positions = snapshot.get("agents", {})

    first_non_attacker: Optional[int] = None
    for idx, candidate in enumerate(candidates, start=1):
        if candidate.is_destination:
            return idx
        if candidate.drone_id not in attackers and first_non_attacker is None:
            first_non_attacker = idx
        if candidate.drone_id in attackers:
            continue
        if _candidate_inside_jammer(candidate.drone_id, jammer, positions):
            continue
        return idx

    if first_non_attacker is not None:
        return first_non_attacker
    return 0


def _candidate_inside_jammer(drone_id: int, jammer: Optional[dict], positions: dict) -> bool:
    if not jammer or not jammer.get("active"):
        return False
    coords = positions.get(f"uav_{drone_id}", {}).get("position")
    if not coords:
        return False
    jammer_pos = jammer.get("position", (0.0, 0.0, 0.0))
    radius = float(jammer.get("radius", 0.0))
    return _distance(coords, jammer_pos) <= radius


def summarize_frames(frames: list[dict]) -> dict:
    if not frames:
        return {}
    final_metrics = frames[-1].get("metrics", {})
    time_s = max(float(final_metrics.get("time", frames[-1].get("time", 0.0))), 1e-9)
    generated = float(final_metrics.get("generated_packets", 0.0))
    delivered = float(final_metrics.get("delivered_packets", 0.0))
    secure_throughput = delivered / time_s
    pdr = float(final_metrics.get("packet_delivery_ratio", 0.0))
    delay_ms = float(final_metrics.get("average_delay_ms", 0.0))
    security_events = float(final_metrics.get("security_event_count", 0.0))
    collisions = float(final_metrics.get("collision_count", 0.0))
    objective = 200.0 * pdr + 10.0 * secure_throughput - 0.02 * delay_ms - 0.001 * security_events - 0.01 * collisions
    return {
        "time": time_s,
        "generated_packets": generated,
        "delivered_packets": delivered,
        "packet_delivery_ratio": pdr,
        "secure_throughput": secure_throughput,
        "average_delay_ms": delay_ms,
        "security_event_count": security_events,
        "collision_count": collisions,
        "ack_timeouts": float(final_metrics.get("ack_timeouts", 0.0)),
        "route_decisions": float(final_metrics.get("route_decisions", 0.0)),
        "mean_residual_energy": float(final_metrics.get("mean_residual_energy", 0.0)),
        "objective": objective,
    }


def render_html(runs: list[dict], config: UAVSim4SecurityEnvConfig, args: argparse.Namespace) -> str:
    data = {
        "runs": runs,
        "meta": {
            "mode": args.mode,
            "attacks": list(config.attacks),
            "attacker_ids": list(config.attacker_ids),
            "episode_duration": config.episode_duration,
            "control_interval": config.control_interval,
            "num_uavs": config.num_uavs,
            "candidate_limit": config.candidate_limit,
            "sensing_range": config.sensing_range,
            "seed": args.seed,
        },
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA_JSON__", payload)


def _format_summary(policy: str, summary: dict) -> str:
    return (
        f"{policy}: generated={summary.get('generated_packets', 0):.0f}, "
        f"delivered={summary.get('delivered_packets', 0):.0f}, "
        f"pdr={summary.get('packet_delivery_ratio', 0):.3f}, "
        f"throughput={summary.get('secure_throughput', 0):.2f} pkt/s, "
        f"delay_ms={summary.get('average_delay_ms', 0):.2f}, "
        f"security_events={summary.get('security_event_count', 0):.0f}, "
        f"objective={summary.get('objective', 0):.2f}"
    )


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_coords(value: str) -> Optional[tuple[float, float, float]]:
    if not value.strip():
        return None
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise ValueError("--jammer-coords must contain x,y,z")
    return (parts[0], parts[1], parts[2])


def _distance(a, b) -> float:
    return math.sqrt(sum((float(a[idx]) - float(b[idx])) ** 2 for idx in range(3)))


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UAVSim4Security Strategy Comparison</title>
<style>
:root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
body { margin: 0; background: #101418; color: #e8edf2; }
.wrap { min-height: 100vh; display: grid; grid-template-rows: auto 1fr auto; }
header { padding: 14px 18px 8px; border-bottom: 1px solid rgba(255,255,255,.09); }
h1 { margin: 0; font-size: 18px; font-weight: 720; letter-spacing: 0; }
.sub { margin-top: 5px; color: #a9b5c0; font-size: 13px; }
main { padding: 14px 18px 10px; }
.stage { max-width: 1280px; margin: 0 auto; }
canvas { display: block; width: 100%; height: auto; background: #141b22; border: 1px solid rgba(255,255,255,.1); border-radius: 8px; }
.controls { display: grid; grid-template-columns: auto 1fr auto auto auto; gap: 10px; align-items: center; margin-top: 10px; }
button, select { background: #202a33; color: #e8edf2; border: 1px solid rgba(255,255,255,.14); border-radius: 6px; height: 34px; padding: 0 12px; font: inherit; }
input[type="range"] { width: 100%; accent-color: #5db7de; }
footer { color: #8c9aa6; font-size: 12px; padding: 0 18px 14px; text-align: center; }
@media (max-width: 740px) { .controls { grid-template-columns: 1fr; } h1 { font-size: 16px; } }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>UAV-MEC Cross-layer Defense Strategy Comparison</h1>
    <div class="sub">Same seed and attack profile, synchronized replay for baseline and proposed defense policies.</div>
  </header>
  <main>
    <div class="stage">
      <canvas id="canvas" width="1280" height="900"></canvas>
      <div class="controls">
        <button id="play">Pause</button>
        <input id="slider" type="range" min="0" max="0" value="0" step="1" aria-label="Frame">
        <select id="speed" aria-label="Speed">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
        <button id="step">Step</button>
        <button id="reset">Reset</button>
      </div>
    </div>
  </main>
  <footer>Generated by showcases/rl-bridge/main_uavsim4security_visualize.py</footer>
</div>
<script>
const DATA = __DATA_JSON__;
const runs = DATA.runs || [];
const meta = DATA.meta || {};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const slider = document.getElementById('slider');
const playButton = document.getElementById('play');
const speedSelect = document.getElementById('speed');
const stepButton = document.getElementById('step');
const resetButton = document.getElementById('reset');
const palette = { greedy: '#5db7de', random: '#f2c94c', defensive: '#5edca8' };
const attackColor = '#ef6c66';
const maxFrames = Math.max(1, ...runs.map(run => (run.frames || []).length));
slider.max = Math.max(0, maxFrames - 1);
let frameIndex = 0;
let playing = true;
let lastTick = performance.now();
let carry = 0;

function frameFor(run, idx = frameIndex) {
  const frames = run.frames || [];
  if (!frames.length) return {};
  return frames[Math.max(0, Math.min(idx, frames.length - 1))] || {};
}
function mapInfo(frame) { return frame.map || { length: 600, width: 600, height: 100 }; }
function sx(rect, frame, x) { const m = mapInfo(frame); return rect.x + (Number(x) / m.length) * rect.w; }
function sy(rect, frame, y) { const m = mapInfo(frame); return rect.y + rect.h - (Number(y) / m.width) * rect.h; }
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function eventTimeSeconds(event) { return Number(event.time !== undefined ? event.time : event.time_us / 1e6); }
function metric(frame, name) { return Number((frame.metrics || {})[name] || 0); }
function finalSummary(run) { return run.summary || {}; }
function secureThroughput(frame) { const t = Math.max(Number(frame.time || 0), 1e-9); return metric(frame, 'delivered_packets') / t; }

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBackground();
  drawHeader();
  const panels = layoutPanels();
  runs.forEach((run, idx) => drawRunPanel(run, panels[idx], idx));
  drawComparisonCharts();
  drawRanking();
  slider.value = frameIndex;
}

function drawBackground() {
  const g = ctx.createLinearGradient(0, 0, 0, canvas.height);
  g.addColorStop(0, '#151d24');
  g.addColorStop(1, '#0f1419');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function drawHeader() {
  ctx.fillStyle = '#e8edf2';
  ctx.font = '700 18px Inter, sans-serif';
  ctx.fillText('Cross-layer defense under blackhole, grayhole, and PHY jamming', 28, 30);
  ctx.fillStyle = '#a9b5c0';
  ctx.font = '12px Inter, sans-serif';
  const attacks = (meta.attacks || []).join(', ') || 'none';
  ctx.fillText(`duration=${Number(meta.episode_duration || 0).toFixed(1)}s  interval=${Number(meta.control_interval || 0).toFixed(2)}s  range=${Number(meta.sensing_range || 0).toFixed(0)}m  seed=${meta.seed}  attacks=${attacks}`, 28, 50);
}

function layoutPanels() {
  const count = Math.max(1, runs.length);
  const gap = 18;
  const margin = 26;
  const w = (canvas.width - margin * 2 - gap * (count - 1)) / count;
  return runs.map((_, idx) => ({ x: margin + idx * (w + gap), y: 72, w, h: 392 }));
}

function drawRunPanel(run, rect, idx) {
  const frame = frameFor(run);
  const color = palette[run.policy] || '#9ac7ff';
  ctx.save();
  ctx.fillStyle = '#17212a';
  roundRect(rect.x, rect.y, rect.w, rect.h, 8, true, false);
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  roundRect(rect.x, rect.y, rect.w, rect.h, 8, false, true);
  const mapRect = { x: rect.x + 12, y: rect.y + 54, w: rect.w - 24, h: 235 };
  drawPolicyHeader(run, rect, color, frame);
  drawGrid(mapRect);
  drawJammer(frame, mapRect);
  drawMecAnchor(frame, mapRect);
  drawTrails(run, frame, mapRect, color);
  drawRoutes(frame, mapRect, color);
  drawAttackEvents(frame, mapRect);
  drawUavs(frame, mapRect);
  drawMetricStrip(run, frame, rect, color);
  ctx.restore();
}

function drawPolicyHeader(run, rect, color, frame) {
  ctx.fillStyle = color;
  ctx.font = '700 15px Inter, sans-serif';
  ctx.fillText(run.title || run.policy, rect.x + 14, rect.y + 24);
  ctx.fillStyle = '#9facb7';
  ctx.font = '11px Inter, sans-serif';
  const desc = run.description || '';
  ctx.fillText(desc.length > 72 ? desc.slice(0, 69) + '...' : desc, rect.x + 14, rect.y + 42);
  ctx.fillStyle = '#e8edf2';
  ctx.font = '700 12px Inter, sans-serif';
  ctx.fillText(`t=${Number(frame.time || 0).toFixed(1)}s`, rect.x + rect.w - 66, rect.y + 24);
}

function drawGrid(rect) {
  ctx.fillStyle = '#101820';
  ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
  ctx.strokeStyle = 'rgba(255,255,255,.07)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const x = rect.x + (rect.w / 4) * i;
    const y = rect.y + (rect.h / 4) * i;
    ctx.beginPath(); ctx.moveTo(x, rect.y); ctx.lineTo(x, rect.y + rect.h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(rect.x, y); ctx.lineTo(rect.x + rect.w, y); ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(255,255,255,.18)';
  ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
}

function drawJammer(frame, rect) {
  const jammer = frame.jammer;
  if (!jammer) return;
  const p = jammer.position || [300, 300, 50];
  const m = mapInfo(frame);
  const radiusPx = Number(jammer.radius || 0) / Number(m.length || 600) * rect.w;
  const x = sx(rect, frame, p[0]);
  const y = sy(rect, frame, p[1]);
  const pulse = 1 + 0.06 * Math.sin(Date.now() / 180);
  ctx.beginPath();
  ctx.arc(x, y, radiusPx * pulse, 0, Math.PI * 2);
  ctx.fillStyle = jammer.active ? 'rgba(239, 83, 80, .18)' : 'rgba(239, 83, 80, .07)';
  ctx.fill();
  ctx.strokeStyle = jammer.active ? 'rgba(239, 83, 80, .82)' : 'rgba(239, 83, 80, .35)';
  ctx.setLineDash([7, 6]);
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
  drawLabel('JAM', x + 8, y - 8, '#ff9d96');
}

function drawMecAnchor(frame, rect) {
  const m = mapInfo(frame);
  const x = sx(rect, frame, m.length / 2);
  const y = sy(rect, frame, m.width / 2);
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(Math.PI / 4);
  ctx.fillStyle = '#f2c94c';
  ctx.strokeStyle = 'rgba(0,0,0,.45)';
  ctx.lineWidth = 1.5;
  ctx.fillRect(-7, -7, 14, 14);
  ctx.strokeRect(-7, -7, 14, 14);
  ctx.restore();
  drawLabel('MEC', x + 9, y + 4, '#f7d36c');
}

function drawTrails(run, frame, rect, color) {
  const frames = run.frames || [];
  const start = Math.max(0, frameIndex - 14);
  for (let i = start; i < Math.min(frameIndex, frames.length - 1); i++) {
    const a = frames[i];
    const b = frames[i + 1];
    if (!a || !b || !a.agent_positions || !b.agent_positions) continue;
    const alpha = (i - start + 1) / Math.max(1, frameIndex - start + 1) * 0.24;
    for (const agent of Object.keys(b.agent_positions)) {
      const p1 = a.agent_positions[agent];
      const p2 = b.agent_positions[agent];
      if (!p1 || !p2) continue;
      ctx.strokeStyle = rgba(color, alpha);
      ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.moveTo(sx(rect, frame, p1[0]), sy(rect, frame, p1[1])); ctx.lineTo(sx(rect, frame, p2[0]), sy(rect, frame, p2[1])); ctx.stroke();
    }
  }
}

function drawRoutes(frame, rect, color) {
  const positions = frame.agent_positions || {};
  const now = Number(frame.time || 0);
  const decisions = frame.recent_route_decisions || [];
  for (const decision of decisions) {
    if (now - Number(decision.time || 0) > 1.2) continue;
    if (decision.next_hop_id === null || decision.next_hop_id === undefined) continue;
    const from = positions[`uav_${decision.drone_id}`];
    const to = positions[`uav_${decision.next_hop_id}`];
    if (!from || !to) continue;
    const age = clamp(1 - (now - Number(decision.time || 0)) / 1.2, 0.12, 1);
    ctx.strokeStyle = rgba(color, 0.28 + 0.58 * age);
    ctx.lineWidth = 2;
    const x1 = sx(rect, frame, from[0]);
    const y1 = sy(rect, frame, from[1]);
    const x2 = sx(rect, frame, to[0]);
    const y2 = sy(rect, frame, to[1]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    drawArrowHead(x1, y1, x2, y2, color, age);
  }
}

function drawArrowHead(x1, y1, x2, y2, color, alpha) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const size = 6;
  ctx.save();
  ctx.translate(x2, y2);
  ctx.rotate(angle);
  ctx.fillStyle = rgba(color, alpha);
  ctx.beginPath();
  ctx.moveTo(0, 0); ctx.lineTo(-size, -size / 2); ctx.lineTo(-size, size / 2); ctx.closePath();
  ctx.fill();
  ctx.restore();
}

function drawAttackEvents(frame, rect) {
  const events = frame.attack_events || [];
  const positions = frame.agent_positions || {};
  const now = Number(frame.time || 0);
  for (const event of events) {
    const t = eventTimeSeconds(event);
    if (now - t > 1.6) continue;
    const p = positions[`uav_${event.drone_id}`];
    if (!p) continue;
    const age = clamp(1 - (now - t) / 1.6, 0.1, 1);
    const x = sx(rect, frame, p[0]);
    const y = sy(rect, frame, p[1]);
    ctx.strokeStyle = rgba(attackColor, age);
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.moveTo(x - 9, y - 9); ctx.lineTo(x + 9, y + 9); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + 9, y - 9); ctx.lineTo(x - 9, y + 9); ctx.stroke();
  }
}

function drawUavs(frame, rect) {
  const agents = frame.agents || {};
  const positions = frame.agent_positions || {};
  for (const [agent, pos] of Object.entries(positions)) {
    const state = agents[agent] || {};
    const x = sx(rect, frame, pos[0]);
    const y = sy(rect, frame, pos[1]);
    const attacker = Boolean(state.is_attacker);
    ctx.beginPath();
    ctx.arc(x, y, attacker ? 8.5 : 7.5, 0, Math.PI * 2);
    ctx.fillStyle = attacker ? attackColor : '#d3f3ff';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = attacker ? '#ffd0cd' : '#5db7de';
    ctx.stroke();
    const q = Number(state.queue_length || 0);
    if (q > 0) {
      ctx.beginPath();
      ctx.arc(x, y, 11 + Math.min(q, 7), 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,.22)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    drawLabel(agent.replace('uav_', 'U'), x + 9, y - 8, attacker ? '#ffd0cd' : '#d3f3ff');
  }
}

function drawMetricStrip(run, frame, rect, color) {
  const m = frame.metrics || {};
  const y = rect.y + rect.h - 78;
  const cols = [
    ['PDR', Number(m.packet_delivery_ratio || 0).toFixed(3)],
    ['Thr.', secureThroughput(frame).toFixed(2)],
    ['Delay', Number(m.average_delay_ms || 0).toFixed(1)],
    ['Sec.', Number(m.security_event_count || 0).toFixed(0)],
  ];
  ctx.font = '11px Inter, sans-serif';
  cols.forEach((item, idx) => {
    const w = (rect.w - 28) / cols.length;
    const x = rect.x + 14 + idx * w;
    ctx.fillStyle = '#101820';
    roundRect(x, y, w - 7, 52, 6, true, false);
    ctx.fillStyle = '#8c9aa6';
    ctx.fillText(item[0], x + 8, y + 18);
    ctx.fillStyle = idx === 0 ? color : '#e8edf2';
    ctx.font = '700 14px Inter, sans-serif';
    ctx.fillText(item[1], x + 8, y + 39);
    ctx.font = '11px Inter, sans-serif';
  });
}

function drawComparisonCharts() {
  const area = { x: 26, y: 490, w: 820, h: 380 };
  ctx.fillStyle = '#17212a';
  roundRect(area.x, area.y, area.w, area.h, 8, true, false);
  ctx.strokeStyle = 'rgba(255,255,255,.12)';
  roundRect(area.x, area.y, area.w, area.h, 8, false, true);
  ctx.fillStyle = '#e8edf2';
  ctx.font = '700 15px Inter, sans-serif';
  ctx.fillText('Synchronized metric trends', area.x + 16, area.y + 26);
  const charts = [
    { label: 'Packet delivery ratio', getter: f => metric(f, 'packet_delivery_ratio'), max: 1 },
    { label: 'Secure throughput (pkt/s)', getter: f => secureThroughput(f), max: maxAcross(f => secureThroughput(f)) },
    { label: 'Average E2E delay (ms)', getter: f => metric(f, 'average_delay_ms'), max: maxAcross(f => metric(f, 'average_delay_ms')) },
    { label: 'Security events', getter: f => metric(f, 'security_event_count'), max: maxAcross(f => metric(f, 'security_event_count')) },
  ];
  charts.forEach((chart, idx) => {
    const col = idx % 2;
    const row = Math.floor(idx / 2);
    const rect = { x: area.x + 18 + col * 395, y: area.y + 48 + row * 156, w: 370, h: 132 };
    drawChart(rect, chart.label, chart.getter, Math.max(1e-9, chart.max));
  });
}

function drawChart(rect, label, getter, maxValue) {
  ctx.fillStyle = '#101820';
  roundRect(rect.x, rect.y, rect.w, rect.h, 6, true, false);
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  roundRect(rect.x, rect.y, rect.w, rect.h, 6, false, true);
  ctx.fillStyle = '#9facb7';
  ctx.font = '12px Inter, sans-serif';
  ctx.fillText(label, rect.x + 10, rect.y + 18);
  const plot = { x: rect.x + 12, y: rect.y + 30, w: rect.w - 24, h: rect.h - 45 };
  ctx.strokeStyle = 'rgba(255,255,255,.08)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = plot.y + (plot.h / 3) * i;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
  }
  for (const run of runs) {
    const frames = run.frames || [];
    const color = palette[run.policy] || '#9ac7ff';
    ctx.strokeStyle = color;
    ctx.lineWidth = run.policy === 'defensive' ? 2.4 : 1.8;
    ctx.beginPath();
    const count = Math.min(frameIndex + 1, frames.length);
    for (let i = 0; i < count; i++) {
      const px = plot.x + (i / Math.max(1, maxFrames - 1)) * plot.w;
      const py = plot.y + plot.h - clamp(getter(frames[i]) / maxValue, 0, 1) * plot.h;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
}

function drawRanking() {
  const rect = { x: 872, y: 490, w: 382, h: 380 };
  ctx.fillStyle = '#17212a';
  roundRect(rect.x, rect.y, rect.w, rect.h, 8, true, false);
  ctx.strokeStyle = 'rgba(255,255,255,.12)';
  roundRect(rect.x, rect.y, rect.w, rect.h, 8, false, true);
  ctx.fillStyle = '#e8edf2';
  ctx.font = '700 15px Inter, sans-serif';
  ctx.fillText('Final comparison', rect.x + 16, rect.y + 26);
  const sorted = [...runs].sort((a, b) => Number((b.summary || {}).objective || 0) - Number((a.summary || {}).objective || 0));
  let y = rect.y + 58;
  for (const run of sorted) {
    const s = finalSummary(run);
    const color = palette[run.policy] || '#9ac7ff';
    ctx.fillStyle = '#101820';
    roundRect(rect.x + 16, y - 22, rect.w - 32, 78, 6, true, false);
    ctx.fillStyle = color;
    ctx.font = '700 13px Inter, sans-serif';
    ctx.fillText(run.title || run.policy, rect.x + 28, y);
    ctx.fillStyle = '#e8edf2';
    ctx.font = '700 12px Inter, sans-serif';
    ctx.fillText(`score ${Number(s.objective || 0).toFixed(1)}`, rect.x + rect.w - 92, y);
    ctx.fillStyle = '#9facb7';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(`PDR ${Number(s.packet_delivery_ratio || 0).toFixed(3)}   thr ${Number(s.secure_throughput || 0).toFixed(2)} pkt/s`, rect.x + 28, y + 22);
    ctx.fillText(`delay ${Number(s.average_delay_ms || 0).toFixed(1)} ms   sec ${Number(s.security_event_count || 0).toFixed(0)}   coll ${Number(s.collision_count || 0).toFixed(0)}`, rect.x + 28, y + 42);
    y += 90;
  }
  drawLegend(rect.x + 16, rect.y + rect.h - 50);
}

function drawLegend(x, y) {
  const items = [
    ['#d3f3ff', 'normal UAV'],
    [attackColor, 'attacker/event'],
    ['#f2c94c', 'MEC/JAM marker'],
  ];
  ctx.font = '11px Inter, sans-serif';
  items.forEach((item, idx) => {
    ctx.fillStyle = item[0];
    ctx.fillRect(x + idx * 118, y - 10, 10, 10);
    ctx.fillStyle = '#b9c4cc';
    ctx.fillText(item[1], x + 15 + idx * 118, y);
  });
}

function maxAcross(getter) {
  let maxValue = 0;
  for (const run of runs) {
    for (const frame of run.frames || []) maxValue = Math.max(maxValue, Number(getter(frame) || 0));
  }
  return maxValue || 1;
}

function drawLabel(text, x, y, color) {
  ctx.font = '700 10px Inter, sans-serif';
  ctx.fillStyle = 'rgba(0,0,0,.55)';
  ctx.fillText(text, x + 1, y + 1);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
}

function roundRect(x, y, w, h, r, fill, stroke) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

function rgba(hex, alpha) {
  const cleaned = hex.replace('#', '');
  const value = parseInt(cleaned, 16);
  const r = (value >> 16) & 255;
  const g = (value >> 8) & 255;
  const b = value & 255;
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function tick(now) {
  if (playing && maxFrames > 1) {
    const speed = Number(speedSelect.value || 1);
    carry += (now - lastTick) / 540 * speed;
    while (carry >= 1) {
      frameIndex = (frameIndex + 1) % maxFrames;
      carry -= 1;
    }
  }
  lastTick = now;
  draw();
  requestAnimationFrame(tick);
}

playButton.addEventListener('click', () => {
  playing = !playing;
  playButton.textContent = playing ? 'Pause' : 'Play';
});
slider.addEventListener('input', () => {
  frameIndex = Number(slider.value);
  playing = false;
  playButton.textContent = 'Play';
});
stepButton.addEventListener('click', () => {
  frameIndex = Math.min(maxFrames - 1, frameIndex + 1);
  playing = false;
  playButton.textContent = 'Play';
});
resetButton.addEventListener('click', () => {
  frameIndex = 0;
  carry = 0;
  playing = false;
  playButton.textContent = 'Play';
});
requestAnimationFrame(tick);
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
