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
        sd = stddev(vals)
        ys_mean.append(mu)
        ys_low.append(mu - sd)
        ys_high.append(mu + sd)

    return xs, ys_mean, ys_low, ys_high


def plot_multiple_files(pattern="ilp_times-*.txt", output_file="ilp_times_multi.png"):
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
        print("NodeCount\tSamples\tMean\tStdDev")
        for x in xs:
            vals = data[x]
            print(f"{x}\t{len(vals)}\t{mean(vals):.6f}\t{stddev(vals):.6f}")

    plt.xlabel("Number of nodes")
    plt.ylabel("Execution time (s)")
    plt.title("Average ILP execution time per file")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_file}")
    plt.show()


if __name__ == "__main__":
    plot_multiple_files(pattern="ilp_times-*.txt", output_file="ilp_times_multi.png")