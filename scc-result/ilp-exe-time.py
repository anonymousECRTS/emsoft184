import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

filename = "scc_times-1.txt"
pattern = r"\./scc-500/(\d+)/benchmark_(\d+)/benchmark\.json: \[([0-9.]+)\]"

data = defaultdict(list)

with open(filename, "r") as f:
    for line in f:
        m = re.match(pattern, line.strip())
        if m:
            group = int(m.group(1))
            value = float(m.group(3))
            data[group].append(value)

def bootstrap_ci_mean(arr, n_boot=10000, alpha=0.05):
    arr = np.array(arr, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return np.mean(arr), lower, upper

groups = sorted(data.keys())
means = []
ci_low = []
ci_high = []

for g in groups:
    mean, low, high = bootstrap_ci_mean(data[g])
    means.append(mean)
    ci_low.append(low)
    ci_high.append(high)

means = np.array(means)
ci_low = np.array(ci_low)
ci_high = np.array(ci_high)

plt.figure(figsize=(10, 6))
plt.plot(groups, means, marker='o', label='Mean')
plt.fill_between(groups, ci_low, ci_high, alpha=0.25, label='95% bootstrap CI')

plt.xscale("log", base=2)
plt.yscale("log")
plt.xticks(groups, groups)
plt.xlabel("Group")
plt.ylabel("Value")
plt.title("Mean benchmark value with 95% confidence interval")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()