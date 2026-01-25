import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 数据
genome_size = np.array([
    6233, 3494, 6645, 4209, 5552, 7475, 5787, 6021, 5061, 5998,
    5866, 3814, 4882, 4140, 8640, 3662, 3894, 4120, 7194, 4214,
    4237, 4397
])
seconds = np.array([
    1620, 1479, 1496, 1484, 1462, 1528, 1487, 1470, 1429, 1359,
    1599, 1529, 1410, 1506, 1669, 1613, 1568, 1407, 1571, 1396,
    1694, 1611
])
minutes = seconds / 60.0
perprotein = seconds / genome_size
plt.rcParams.update({
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'font.family': 'sans-serif',

    'figure.dpi': 300
})

coef = np.polyfit(genome_size, minutes, 1)
fit_fn = np.poly1d(coef)


plt.figure(figsize=(4, 4))
plt.scatter(genome_size, minutes, label="Samples", color="#7FADDC", marker="o")
plt.plot(genome_size, fit_fn(genome_size), linestyle="-", label="Linear fit")
plt.xlabel("Genome size")
plt.ylabel("Runtime (minutes)")
plt.title("Genome size vs Runtime")
plt.legend()
plt.tight_layout()
# plt.show()

plt.savefig("fig/S2_timecost.png", dpi=300)
plt.savefig("fig/S2_timecost.pdf")
