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

SECURITY_PATTERN = re.compile(r"Security metric \[(.+?)\] is:\s+(\d+)")


def parse_metrics(output_text):
    metrics = {}
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output_text)
        if not match:
            raise RuntimeError(f"Failed to parse metric '{metric_name}'.")

        value = match.group(1)
        if value == "nan":
            metrics[metric_name] = float("nan")
        else:
            metrics[metric_name] = float(value) if "." in value else int(value)

    metrics["security_metrics"] = {
        match.group(1): int(match.group(2))
        for match in SECURITY_PATTERN.finditer(output_text)
    }
    return metrics


def run_scenario(protocol_name, scenario_name, attacks="", attacker_ids="", probability="1.0"):
    env = os.environ.copy()
    env.update({
        "MPLBACKEND": "Agg",
        "UAVNETSIM_ROUTING_PROTOCOL": protocol_name,
        "UAVNETSIM_ENABLE_VISUALIZATION": "0",
        "UAVNETSIM_SEED": env.get("UAVNETSIM_SEED", "2025"),
        "UAVNETSIM_SIM_TIME_US": env.get("UAVNETSIM_SIM_TIME_US", str(int(10 * 1e6))),
        "UAVNETSIM_LOG_FILE": f"running_log_{protocol_name.lower()}_{scenario_name}.log",
    })

    if attacks:
        env["UAVNETSIM_ATTACKS"] = attacks
        env["UAVNETSIM_ATTACKER_IDS"] = attacker_ids
        env["UAVNETSIM_ATTACK_PROBABILITY"] = probability
    else:
        env.pop("UAVNETSIM_ATTACKS", None)
        env.pop("UAVNETSIM_ATTACKER_IDS", None)
        env.pop("UAVNETSIM_ATTACK_PROBABILITY", None)

    stdout_path = RESULTS_DIR / f"{protocol_name.lower()}_{scenario_name}_stdout.txt"
    stderr_path = RESULTS_DIR / f"{protocol_name.lower()}_{scenario_name}_stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        completed = subprocess.run(
            [sys.executable, "main.py"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Scenario '{scenario_name}' failed with exit code {completed.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )

    stdout_text = stdout_path.read_text(encoding="utf-8")

    return {
        "scenario": scenario_name,
        "attacks": attacks.split(",") if attacks else [],
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "log_path": str(PROJECT_ROOT / env["UAVNETSIM_LOG_FILE"]),
        "metrics": parse_metrics(stdout_text),
    }


def build_markdown_report(protocol_name, results):
    lines = [
        f"# Security Comparison for {protocol_name}",
        "",
        "| Scenario | PDR | Delay (ms) | Routing Load | Throughput (Kbps) | Hop Count | Collisions | MAC Delay (ms) | Security Metrics |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for scenario_name, result in results.items():
        metrics = result["metrics"]
        security_metrics = ", ".join(
            f"{key}={value}" for key, value in sorted(metrics["security_metrics"].items())
        ) or "none"
        lines.append(
            f"| {scenario_name} | {metrics['packet_delivery_ratio_percent']} | {metrics['average_end_to_end_delay_ms']} | "
            f"{metrics['routing_load']} | {metrics['average_throughput_kbps']} | {metrics['average_hop_count']} | "
            f"{metrics['collision_count']} | {metrics['average_mac_delay_ms']} | {security_metrics} |"
        )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    protocol_name = os.environ.get("UAVNETSIM_SECURITY_PROTOCOL", "DSDV").upper()
    attacker_ids = os.environ.get("UAVNETSIM_SECURITY_ATTACKER_IDS", "1,3")
    probability = os.environ.get("UAVNETSIM_SECURITY_ATTACK_PROBABILITY", "1.0")

    scenarios = {
        "baseline": {"attacks": "", "attacker_ids": "", "probability": probability},
        "blackhole": {"attacks": "BLACKHOLE", "attacker_ids": attacker_ids, "probability": probability},
        "grayhole": {"attacks": "GRAYHOLE", "attacker_ids": attacker_ids, "probability": probability},
        "ack_spoof": {"attacks": "ACK_SPOOF", "attacker_ids": attacker_ids, "probability": probability},
        "ack_poison": {"attacks": "ACK_POISON", "attacker_ids": attacker_ids, "probability": probability},
        "location_spoof": {"attacks": "LOC_SPOOF", "attacker_ids": attacker_ids, "probability": probability},
    }

    results = {}
    for scenario_name, scenario in scenarios.items():
        print(f"Running {protocol_name} / {scenario_name}...")
        results[scenario_name] = run_scenario(
            protocol_name,
            scenario_name,
            attacks=scenario["attacks"],
            attacker_ids=scenario["attacker_ids"],
            probability=scenario["probability"],
        )

    json_path = RESULTS_DIR / f"security_comparison_{protocol_name.lower()}.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    markdown_path = RESULTS_DIR / f"security_comparison_{protocol_name.lower()}.md"
    markdown_path.write_text(build_markdown_report(protocol_name, results), encoding="utf-8")

    print(json.dumps(results, indent=2))
    print(f"Security comparison report written to {markdown_path}")


if __name__ == "__main__":
    main()
