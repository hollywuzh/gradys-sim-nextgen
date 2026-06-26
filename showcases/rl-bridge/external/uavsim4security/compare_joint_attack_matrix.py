import csv
import json
import os
from pathlib import Path

from compare_security import run_scenario


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_csv_env(name, default):
    value = os.environ.get(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_attacker_groups(name, default):
    value = os.environ.get(name, default)
    return [group.strip() for group in value.split(";") if group.strip()]


def build_scenarios(probability, attacker_ids):
    return {
        "baseline": {"attacks": "", "attacker_ids": "", "probability": probability},
        "ack_poison": {"attacks": "ACK_POISON", "attacker_ids": attacker_ids, "probability": probability},
        "grayhole": {"attacks": "GRAYHOLE", "attacker_ids": attacker_ids, "probability": probability},
        "ack_poison_grayhole": {
            "attacks": "ACK_POISON,GRAYHOLE",
            "attacker_ids": attacker_ids,
            "probability": probability,
        },
    }


def flatten_security_metrics(metrics):
    security_metrics = metrics.get("security_metrics", {})
    if not security_metrics:
        return "none"

    return ", ".join(f"{key}={value}" for key, value in sorted(security_metrics.items()))


def collect_rows(results):
    rows = []
    for protocol_name, probability_results in results.items():
        for probability, attacker_group_results in probability_results.items():
            for attacker_group, scenario_results in attacker_group_results.items():
                baseline_metrics = scenario_results["baseline"]["metrics"]
                for scenario_name, result in scenario_results.items():
                    metrics = result["metrics"]
                    rows.append({
                        "protocol": protocol_name,
                        "attack_probability": probability,
                        "attacker_ids": attacker_group,
                        "scenario": scenario_name,
                        "generated_packets": metrics["generated_packets"],
                        "packet_delivery_ratio_percent": metrics["packet_delivery_ratio_percent"],
                        "average_end_to_end_delay_ms": metrics["average_end_to_end_delay_ms"],
                        "routing_load": metrics["routing_load"],
                        "average_throughput_kbps": metrics["average_throughput_kbps"],
                        "average_hop_count": metrics["average_hop_count"],
                        "collision_count": metrics["collision_count"],
                        "average_mac_delay_ms": metrics["average_mac_delay_ms"],
                        "security_metrics": flatten_security_metrics(metrics),
                        "pdr_delta_vs_baseline": metrics["packet_delivery_ratio_percent"] - baseline_metrics["packet_delivery_ratio_percent"],
                        "delay_delta_vs_baseline_ms": metrics["average_end_to_end_delay_ms"] - baseline_metrics["average_end_to_end_delay_ms"],
                        "throughput_delta_vs_baseline_kbps": metrics["average_throughput_kbps"] - baseline_metrics["average_throughput_kbps"],
                    })
    return rows


def write_csv(rows, path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(rows):
    lines = [
        "# Joint Attack Matrix",
        "",
        "| Protocol | Probability | Attackers | Scenario | PDR | Delay (ms) | Routing Load | Throughput (Kbps) | PDR Delta | Delay Delta (ms) | Security Metrics |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['attack_probability']} | {row['attacker_ids']} | {row['scenario']} | "
            f"{row['packet_delivery_ratio_percent']} | {row['average_end_to_end_delay_ms']} | {row['routing_load']} | "
            f"{row['average_throughput_kbps']} | {row['pdr_delta_vs_baseline']} | "
            f"{row['delay_delta_vs_baseline_ms']} | {row['security_metrics']} |"
        )

    return "\n".join(lines) + "\n"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    protocol_names = [item.upper() for item in parse_csv_env(
        "UAVNETSIM_JOINT_PROTOCOLS",
        "QROUTING,QFANET,QMR,QGEO",
    )]
    probabilities = parse_csv_env("UAVNETSIM_JOINT_PROBABILITIES", "0.5,1.0")
    attacker_groups = parse_attacker_groups("UAVNETSIM_JOINT_ATTACKER_GROUPS", "1,3")

    results = {}

    for protocol_name in protocol_names:
        results[protocol_name] = {}
        for probability in probabilities:
            results[protocol_name][probability] = {}
            for attacker_group in attacker_groups:
                scenario_results = {}
                scenarios = build_scenarios(probability, attacker_group)

                for scenario_name, scenario in scenarios.items():
                    run_name = (
                        f"{protocol_name.lower()}_{scenario_name}_"
                        f"p{probability.replace('.', '_')}_"
                        f"a{attacker_group.replace(',', '-')}"
                    )
                    print(
                        f"Running {protocol_name} / {scenario_name} / "
                        f"prob={probability} / attackers={attacker_group}..."
                    )
                    scenario_results[scenario_name] = run_scenario(
                        protocol_name=protocol_name,
                        scenario_name=run_name,
                        attacks=scenario["attacks"],
                        attacker_ids=scenario["attacker_ids"],
                        probability=scenario["probability"],
                    )

                results[protocol_name][probability][attacker_group] = scenario_results

    rows = collect_rows(results)

    json_path = RESULTS_DIR / "joint_attack_matrix.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    csv_path = RESULTS_DIR / "joint_attack_matrix.csv"
    write_csv(rows, csv_path)

    markdown_path = RESULTS_DIR / "joint_attack_matrix.md"
    markdown_path.write_text(build_markdown(rows), encoding="utf-8")

    print(f"Joint attack matrix written to {json_path}")
    print(f"Joint attack matrix written to {csv_path}")
    print(f"Joint attack matrix written to {markdown_path}")


if __name__ == "__main__":
    main()
