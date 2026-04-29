"""Plot powergrid evaluation results (zero-shot vs DeLLMa rank)."""
import argparse
import subprocess
import sys
import json
from pathlib import Path

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def run_evaluate(agent_name, mode, results_path):
    """Run evaluate_dellma.py and capture its output."""
    result = subprocess.run(
        [sys.executable, "evaluate_dellma.py",
         "--agent_name", agent_name,
         "--pref_enum_mode", mode,
         "--results_path", results_path],
        capture_output=True, text=True
    )
    return result.stdout + result.stderr


def parse_output(output):
    """Parse evaluation output to extract Accs, Regrets, All-Acc, All-Regret."""
    accs, regrets, all_acc, all_regret = None, None, None, None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("Accs "):
            accs = json.loads(line.split("Accs ")[1])
        elif line.startswith("All-Acc "):
            all_acc = float(line.split("All-Acc ")[1])
        elif line.startswith("Regret") and "opt_mean_stab" in line:
            regrets = json.loads(line.split("] ")[1]) if "] " in line else None
            # format: "Regret (opt_mean_stab - pred_mean_stab) [...]"
            bracket = line[line.index("["):]
            regrets = json.loads(bracket)
        elif line.startswith("All-Regret") and "opt_mean_stab" in line:
            all_regret = float(line.rsplit(" ", 1)[-1])
    return accs, regrets, all_acc, all_regret


def plot_results(accs_zs, reg_zs, all_acc_zs, all_reg_zs,
                 accs_rank, reg_rank, all_acc_rank, all_reg_rank,
                 output_path):
    ks = list(range(2, 2 + len(accs_zs)))
    x = range(len(ks))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Accuracy bar chart ---
    bars1 = ax1.bar([i - width/2 for i in x], accs_zs,  width, label='Zero-shot', color='#7b8db1')
    bars2 = ax1.bar([i + width/2 for i in x], accs_rank, width, label='DeLLMa rank', color='#2db391')
    ax1.set_xlabel('Choice-set size')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(
        'Powergrid Evaluation: Zero-shot vs DeLLMa Rank\n'
        f'Overall: All-Acc {all_acc_zs:.1f}% -> {all_acc_rank:.1f}%; '
        f'All-Regret {all_reg_zs:.6f} -> {all_reg_rank:.6f}',
        fontsize=10
    )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([str(k) for k in ks])
    ax1.legend()

    # --- Regret line chart ---
    ax2.plot(ks, reg_zs,  marker='o', color='#7b8db1', label='Zero-shot')
    ax2.plot(ks, reg_rank, marker='o', color='#2db391', label='DeLLMa rank')
    ax2.set_xlabel('Choice-set size')
    ax2.set_ylabel('Regret (lower is better)')
    ax2.legend()
    ax2.set_xticks(ks)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_name", default="powergrid")
    parser.add_argument("--results_path", default="results")
    parser.add_argument("--model_tag", default=None,
                        help="Model subfolder under results_path (e.g. gpt-4o). "
                             "If omitted, results_path is used directly.")
    parser.add_argument("--output", default="powergrid_zero_shot_vs_rank.png")
    args = parser.parse_args()

    effective_path = (
        os.path.join(args.results_path, args.model_tag)
        if args.model_tag else args.results_path
    )

    print(f"Evaluating zero-shot from {effective_path}...")
    out_zs = run_evaluate(args.agent_name, "zero-shot", effective_path)
    accs_zs, reg_zs, all_acc_zs, all_reg_zs = parse_output(out_zs)

    print(f"Evaluating rank from {effective_path}...")
    out_rank = run_evaluate(args.agent_name, "rank", effective_path)
    accs_rank, reg_rank, all_acc_rank, all_reg_rank = parse_output(out_rank)

    print(f"Zero-shot  -> Acc={accs_zs}, Regret={reg_zs}, All-Acc={all_acc_zs}, All-Regret={all_reg_zs}")
    print(f"Rank       -> Acc={accs_rank}, Regret={reg_rank}, All-Acc={all_acc_rank}, All-Regret={all_reg_rank}")

    plot_results(accs_zs, reg_zs, all_acc_zs, all_reg_zs,
                 accs_rank, reg_rank, all_acc_rank, all_reg_rank,
                 args.output)


if __name__ == "__main__":
    main()
