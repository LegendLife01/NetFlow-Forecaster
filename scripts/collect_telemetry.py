"""Collect telemetry from a ContainerLab spine-leaf fabric.

Run this script from the host where Docker/ContainerLab is available. It reads
interface counters with `docker exec`, measures latency/loss with ping, and
writes rows compatible with the LSTM pipeline. If live containers are missing,
it can fall back to realistic simulation for local development.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


DEFAULT_NODES = {
    "clab-ai-traffic-lab-spine1": "10.255.255.1",
    "clab-ai-traffic-lab-spine2": "10.255.255.2",
    "clab-ai-traffic-lab-leaf1": "10.255.0.1",
    "clab-ai-traffic-lab-leaf2": "10.255.0.2",
    "clab-ai-traffic-lab-leaf3": "10.255.0.3",
    "clab-ai-traffic-lab-leaf4": "10.255.0.4",
}
FEATURE_HEADER = ["timestamp", "traffic_mbps", "latency_ms", "packet_loss_pct", "source"]


def run_cmd(cmd: list[str], timeout: int = 8) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if result.returncode == 0:
            return result.stdout
    except Exception:
        return None
    return None


def live_nodes_available(nodes: dict[str, str]) -> bool:
    out = run_cmd(["docker", "ps", "--format", "{{.Names}}"], timeout=5)
    if not out:
        return False
    running = set(out.splitlines())
    return any(node in running for node in nodes)


def get_interface_bytes(container: str, interfaces: tuple[str, ...] = ("eth1", "eth2", "eth3", "eth4", "eth5")) -> int | None:
    out = run_cmd(["docker", "exec", container, "cat", "/proc/net/dev"], timeout=5)
    if not out:
        return None
    total = 0
    found = False
    for line in out.splitlines():
        if ":" not in line:
            continue
        iface, values = line.split(":", 1)
        iface = iface.strip()
        if iface not in interfaces:
            continue
        parts = values.split()
        if len(parts) >= 16:
            rx_bytes = int(parts[0])
            tx_bytes = int(parts[8])
            total += rx_bytes + tx_bytes
            found = True
    return total if found else None


def ping_from(container: str, target_ip: str, count: int = 5) -> tuple[float | None, float | None]:
    out = run_cmd(["docker", "exec", container, "ping", "-c", str(count), "-W", "1", target_ip], timeout=12)
    if not out:
        return None, None

    loss = None
    loss_match = re.search(r"(\d+(?:\.\d+)?)%\s*packet loss", out)
    if loss_match:
        loss = float(loss_match.group(1))

    avg = None
    rtt_match = re.search(r"(?:rtt|round-trip).*?=\s*([\d.]+)/([\d.]+)/", out)
    if rtt_match:
        avg = float(rtt_match.group(2))
    return avg, loss


def generate_probe_traffic(nodes: dict[str, str], count: int) -> None:
    leaves = [name for name in nodes if "-leaf" in name]
    if len(leaves) < 2:
        return
    source = random.choice(leaves)
    targets = [nodes[name] for name in leaves if name != source]
    target = random.choice(targets)
    run_cmd(["docker", "exec", source, "ping", "-c", str(count), "-W", "1", target], timeout=max(8, count + 4))


def live_sample(nodes: dict[str, str], interval: int, probe_count: int, previous: dict[str, int]) -> tuple[float, float, float]:
    before = {node: get_interface_bytes(node) for node in nodes}
    if probe_count > 0:
        generate_probe_traffic(nodes, probe_count)
    time.sleep(interval)
    after = {node: get_interface_bytes(node) for node in nodes}

    deltas = []
    for node, current in after.items():
        base = before.get(node)
        if current is None:
            continue
        if base is None:
            base = previous.get(node, current)
        previous[node] = current
        deltas.append(max(0, current - base))
    traffic_mbps = (sum(deltas) * 8) / max(1, interval * 1_000_000)

    latencies, losses = [], []
    reference = nodes.get("clab-ai-traffic-lab-spine1", "10.255.255.1")
    for node in nodes:
        if node.endswith("spine1"):
            continue
        latency, loss = ping_from(node, reference)
        if latency is not None:
            latencies.append(latency)
        if loss is not None:
            losses.append(loss)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0
    return round(traffic_mbps, 3), round(avg_latency, 3), round(avg_loss, 3)


def simulated_sample(tick: int) -> tuple[float, float, float]:
    hour = tick % 24
    day = (tick // 24) % 7
    business = 50 + 38 * math.sin((hour - 7) * math.pi / 13) if 7 <= hour <= 20 else 12
    weekly = 0.55 if day >= 5 else 1.0
    traffic = max(0.0, business * weekly + random.gauss(0, 5))
    if random.random() < 0.04:
        traffic += random.uniform(60, 130)
    latency = max(0.5, 2.0 + (traffic / 150.0) * 8 + random.gauss(0, 0.35))
    loss = max(0.0, random.gauss(0.08, 0.04))
    if traffic > 90:
        loss += random.uniform(0.4, 2.5)
    if traffic > 120:
        latency += random.uniform(5, 18)
        loss += random.uniform(1, 6)
    return round(traffic, 3), round(latency, 3), round(min(loss, 100), 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=None, help="CSV output path. Overrides --output-root.")
    parser.add_argument("--output-root", default="runs", help="Root folder for timestamped collection folders.")
    parser.add_argument("--run-name", default=None, help="Optional collection folder name.")
    parser.add_argument("--interval", type=int, default=60, help="Seconds between live samples.")
    parser.add_argument("--samples", type=int, default=0, help="Samples to collect; 0 runs forever.")
    parser.add_argument("--mode", choices=["auto", "live", "simulate"], default="auto", help="Telemetry source.")
    parser.add_argument("--probe-count", type=int, default=20, help="Ping packets per sample to create live traffic.")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of replacing it.")
    return parser.parse_args()


def collect() -> None:
    args = parse_args()
    if args.output:
        output = Path(args.output)
        run_dir = output.parent
    else:
        run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_root) / run_name
        output = run_dir / "telemetry.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    use_live = args.mode == "live" or (args.mode == "auto" and live_nodes_available(DEFAULT_NODES))
    if args.mode == "live" and not live_nodes_available(DEFAULT_NODES):
        raise RuntimeError("ContainerLab Docker containers were not found. Deploy the lab first.")

    mode_name = "containerlab" if use_live else "simulation"
    print(f"Telemetry mode: {mode_name}")
    print(f"Run folder: {run_dir}")
    print(f"Writing: {output}")

    file_mode = "a" if args.append else "w"
    previous_bytes: dict[str, int] = {}
    tick = 0
    with output.open(file_mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if file_mode == "w" or output.stat().st_size == 0:
            writer.writerow(FEATURE_HEADER)
        while args.samples == 0 or tick < args.samples:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if use_live:
                traffic, latency, loss = live_sample(DEFAULT_NODES, args.interval, args.probe_count, previous_bytes)
            else:
                traffic, latency, loss = simulated_sample(tick)
                time.sleep(max(0, min(args.interval, 1)))
            writer.writerow([timestamp, traffic, latency, loss, mode_name])
            handle.flush()
            tick += 1
            print(
                f"{timestamp} sample={tick:04d} traffic={traffic:8.3f} Mbps "
                f"latency={latency:7.3f} ms loss={loss:6.3f}%"
            )


if __name__ == "__main__":
    collect()
