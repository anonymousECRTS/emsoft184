import re
import glob
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def stddev(values):
    if not values:
        return float("nan")
    if len(values) == 1:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(var)


def ci95(values):
    """
    Returns (lower, upper) 95% confidence interval for the mean
    using normal approximation: mean ± 1.96 * sd / sqrt(n)
    """
    if not values:
        return float("nan"), float("nan")

    mu = mean(values)
    n = len(values)

    if n == 1:
        return mu, mu

    sd = stddev(values)
    margin = 1.96 * sd / math.sqrt(n)
    return mu - margin, mu + margin


def parse_one_file(filepath):
    """
    Parse one file containing lines like:
    ./deadlock-01/300/benchmark_0042/benchmark.json: [3.800580]

    Returns:
        data[node_count] = [times...]
    """
    data = defaultdict(list)

    line_re = re.compile(
        r"/deadlock-\d+/(\d+)/benchmark_[^:]+:\s*\[([0-9]*\.?[0-9]+)\]"
    )

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = line_re.search(line)
            if not m:
                continue

            node_count = int(m.group(1))
            exec_time = float(m.group(2))
            data[node_count].append(exec_time)

    return data


def summarize_file_data(data):
    xs = sorted(data.keys())
    ys_mean = []
    ys_low = []
    ys_high = []

    for x in xs:
        vals = data[x]
        mu = mean(vals)
        low, high = ci95(vals)
        ys_mean.append(mu)
        ys_low.append(low)
        ys_high.append(high)

    return xs, ys_mean, ys_low, ys_high


def plot_multiple_files(pattern="deadlock-*.txt", output_file="deadlock_multi.png"):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for pattern: {pattern}")
        return

    plt.figure(figsize=(9, 6))

    for filepath in files:
        data = parse_one_file(filepath)
        if not data:
            print(f"Skipping empty/unmatched file: {filepath}")
            continue

        xs, ys_mean, ys_low, ys_high = summarize_file_data(data)
        label = os.path.splitext(os.path.basename(filepath))[0]

        plt.plot(xs, ys_mean, marker="o", linewidth=2, label=label)
        plt.fill_between(xs, ys_low, ys_high, alpha=0.20)

        print(f"\nFile: {filepath}")
        print("NodeCount\tSamples\tMean\tStdDev\tCI95_low\tCI95_high")
        for x in xs:
            vals = data[x]
            low, high = ci95(vals)
            print(
                f"{x}\t{len(vals)}\t{mean(vals):.6f}\t{stddev(vals):.6f}\t{low:.6f}\t{high:.6f}"
            )

    plt.xlabel("Number of nodes", fontsize=18)
    plt.ylabel("Runtime (s)", fontsize=18)
    #plt.title("Average ILP execution time per file with 95% CI")
    plt.grid(True, linestyle=":")
    plt.xticks([10] + list(range(100, 1001, 100)), fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_multiple_files(pattern="Inter-SCC-edge-prob-*.txt", output_file="../deadlock_multi.png")