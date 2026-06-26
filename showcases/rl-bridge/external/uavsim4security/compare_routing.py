import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"

METRIC_PATTERNS = {
    "generated_packets": r"Totally send:\s+(\d+)\s+data packets",
    "packet_delivery_ratio_percent": r"Packet delivery ratio is:\s+([0-9.]+|nan)\s+%",
    "average_end_to_end_delay_ms": r"Average end-to-end delay is:\s+([0-9.]+|nan)\s+ms",
    "routing_load": r"Routing load is:\s+([0-9.]+|nan)",
    "average_throughput_kbps": r"Average throughput is:\s+([0-9.]+|nan)\s+Kbps",
    "average_hop_count": r"Average hop count is:\s+([0-9.]+|nan)",
    "collision_count": r"Collision num is:\s+(\d+)",
    "average_mac_delay_ms": r"Average mac delay is:\s+([0-9.]+|nan)\s+ms",
}


def parse_metrics(output_text):
    metrics = {}

    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output_text)
        if not match:
            raise RuntimeError(f"Failed to parse metric '{metric_name}' from simulation output.")

        value = match.group(1)
        if value == "nan":
            metrics[metric_name] = float("nan")
        else:
            metrics[metric_name] = float(value) if "." in value else int(value)

    return metrics


def run_protocol(protocol_name):
    protocol_lower = protocol_name.lower()
    log_file = f"running_log_{protocol_lower}.log"

    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "UAVNETSIM_ROUTING_PROTOCOL": protocol_name,
        "UAVNETSIM_ENABLE_VISUALIZATION": "0",
        "UAVNETSIM_LOG_FILE": log_file,
        "UAVNETSIM_SEED": "2025",
    })

    completed = subprocess.run(
        [sys.executable, "main.py"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    stdout_path = RESULTS_DIR / f"{protocol_lower}_stdout.txt"
    stdout_path.write_text(completed.stdout, encoding="utf-8")

    stderr_path = RESULTS_DIR / f"{protocol_lower}_stderr.txt"
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    return {
        "protocol": protocol_name,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "log_path": str(PROJECT_ROOT / log_file),
        "metrics": parse_metrics(completed.stdout),
    }


def build_markdown_report(results):
    dsdv_metrics = results["DSDV"]["metrics"]
    grad_metrics = results["GRAD"]["metrics"]
    greedy_metrics = results["GREEDY"]["metrics"]
    opar_metrics = results["OPAR"]["metrics"]
    qgeo_metrics = results["QGEO"]["metrics"]
    qrouting_metrics = results["QROUTING"]["metrics"]
    qfanet_metrics = results["QFANET"]["metrics"]
    qmr_metrics = results["QMR"]["metrics"]

    lines = [
        "# Routing Comparison",
        "",
        "| Metric | DSDV | GRAd | Greedy | OPAR | QGeo | Q-routing | qfanet | qmr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for metric_name in METRIC_PATTERNS:
        lines.append(
            f"| {metric_name} | {dsdv_metrics[metric_name]} | {grad_metrics[metric_name]} | {greedy_metrics[metric_name]} | {opar_metrics[metric_name]} | {qgeo_metrics[metric_name]} | {qrouting_metrics[metric_name]} | {qfanet_metrics[metric_name]} | {qmr_metrics[metric_name]} |"
        )

    lines.extend([
        "",
        "## Summary",
        "",
        f"- PDR delta (GRAd - DSDV): {grad_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- PDR delta (Greedy - DSDV): {greedy_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- PDR delta (OPAR - DSDV): {opar_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- PDR delta (QGeo - DSDV): {qgeo_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- PDR delta (Q-routing - DSDV): {qrouting_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- PDR delta (qfanet - DSDV): {qfanet_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- PDR delta (qmr - DSDV): {qmr_metrics['packet_delivery_ratio_percent'] - dsdv_metrics['packet_delivery_ratio_percent']:.4f} %",
        f"- Delay delta (GRAd - DSDV): {grad_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Delay delta (Greedy - DSDV): {greedy_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Delay delta (OPAR - DSDV): {opar_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Delay delta (QGeo - DSDV): {qgeo_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Delay delta (Q-routing - DSDV): {qrouting_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Delay delta (qfanet - DSDV): {qfanet_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Delay delta (qmr - DSDV): {qmr_metrics['average_end_to_end_delay_ms'] - dsdv_metrics['average_end_to_end_delay_ms']:.4f} ms",
        f"- Throughput delta (GRAd - DSDV): {grad_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Throughput delta (Greedy - DSDV): {greedy_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Throughput delta (OPAR - DSDV): {opar_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Throughput delta (QGeo - DSDV): {qgeo_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Throughput delta (Q-routing - DSDV): {qrouting_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Throughput delta (qfanet - DSDV): {qfanet_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Throughput delta (qmr - DSDV): {qmr_metrics['average_throughput_kbps'] - dsdv_metrics['average_throughput_kbps']:.4f} Kbps",
        f"- Routing load delta (GRAd - DSDV): {grad_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Routing load delta (Greedy - DSDV): {greedy_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Routing load delta (OPAR - DSDV): {opar_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Routing load delta (QGeo - DSDV): {qgeo_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Routing load delta (Q-routing - DSDV): {qrouting_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Routing load delta (qfanet - DSDV): {qfanet_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Routing load delta (qmr - DSDV): {qmr_metrics['routing_load'] - dsdv_metrics['routing_load']:.4f}",
        f"- Collision delta (GRAd - DSDV): {grad_metrics['collision_count'] - dsdv_metrics['collision_count']}",
        f"- Collision delta (Greedy - DSDV): {greedy_metrics['collision_count'] - dsdv_metrics['collision_count']}",
        f"- Collision delta (OPAR - DSDV): {opar_metrics['collision_count'] - dsdv_metrics['collision_count']}",
        f"- Collision delta (QGeo - DSDV): {qgeo_metrics['collision_count'] - dsdv_metrics['collision_count']}",
        f"- Collision delta (Q-routing - DSDV): {qrouting_metrics['collision_count'] - dsdv_metrics['collision_count']}",
        f"- Collision delta (qfanet - DSDV): {qfanet_metrics['collision_count'] - dsdv_metrics['collision_count']}",
        f"- Collision delta (qmr - DSDV): {qmr_metrics['collision_count'] - dsdv_metrics['collision_count']}",
    ])

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    results = {}
    for protocol_name in ("DSDV", "GRAD", "GREEDY", "OPAR", "QGEO", "QROUTING", "QFANET", "QMR"):
        print(f"Running {protocol_name}...")
        results[protocol_name] = run_protocol(protocol_name)

    json_path = RESULTS_DIR / "routing_comparison.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    markdown_path = RESULTS_DIR / "routing_comparison.md"
    markdown_path.write_text(build_markdown_report(results), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Comparison report written to {markdown_path}")


if __name__ == "__main__":
    main()
