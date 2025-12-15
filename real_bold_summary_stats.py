
import sys, os, time
import numpy as np
import showcase1_ageing as utils
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import resource
import time
import fcntl
from scipy.stats import ks_2samp

# =============================================================
# 1) Load FC and FCD matrices
# =============================================================

FC = np.load("sub-CC110187_empFC.npy")      # shape (N, N)
FCD = np.load("sub-CC110187_empFCD.npy")    # shape (W, W)

print("FC shape:", FC.shape)
print("FCD shape:", FCD.shape)

N = FC.shape[0]
W = FCD.shape[0]


# =============================================================
# 2) Summary statistic: FCD variance over upper triangle
# =============================================================

FCD_upper = FCD[np.triu_indices_from(FCD, k=5)]
FCD_VAR_OV_vect = np.var(FCD_upper)



# =============================================================
# 3) Summary statistic: Homotopic FC
#     Pair i with i + N/2 (assumes symmetric parcellation)
# =============================================================

half = N // 2
homotopic_vals = FC[np.arange(half), np.arange(half) + half]
homotopic_mean = np.mean(homotopic_vals)

# =============================================================
# LOAD REAL BOLD TIME SERIES
# =============================================================

bold = np.loadtxt("ts_TVBSchaeferTian120.txt")  # (T, N)
print("Original BOLD shape:", bold.shape)

# ---- Remove initial transient (same as simulation pipeline: remove first 8 points)
bold_ts = bold[117:117+144, :]  
print("Trimmed BOLD shape:", bold_ts.shape)

T, N = bold_ts.shape
half = N // 2


# =============================================================
# Compute FCD dynamics difference (σ_diff^2)
# =============================================================

# 1) We need sliding-window FC matrices
# utils.compute_fcd already computed the FCD matrix but not the FC windows.
# So compute them here manually.

win_len = 5 # same as in FCD computation
step = 9
T = bold_ts.shape[0]
N = bold_ts.shape[1]
n_windows = (T - win_len) // step + 2
print("Number of windows:", n_windows)
FC_windows = []

for w in range(n_windows):
    segment = bold_ts[w : w + win_len]
    FCw = np.corrcoef(segment.T)
    FC_windows.append(FCw)

FC_windows = np.array(FC_windows)  # shape (W, N, N)

# 2) Full-brain FCD upper triangle squared L2 norm

FCD_full = np.zeros((n_windows, n_windows))
for i in range(n_windows):
    for j in range(n_windows):
        v1 = FC_windows[i][np.triu_indices(N, k=1)]
        v2 = FC_windows[j][np.triu_indices(N, k=1)]
        FCD_full[i, j] = np.corrcoef(v1, v2)[0, 1]

FCD_full_upper = FCD_full[np.triu_indices(n_windows, k=1)]
norm_full = np.sum(FCD_full_upper ** 2)
print("FCD_full=", FCD_full.shape)

# 3) Interhemispheric FCD (i <-> i+N/2)
half = N // 2
lh = np.arange(half)
rh = lh + half

# extract interhemispheric edges for each window
inter_vectors = np.array([FCw[lh, rh] for FCw in FC_windows])

# compute interhemispheric FCD
nW = inter_vectors.shape[0]
FCD_inter = np.zeros((nW, nW))
print("FCD_inter=", FCD_inter.shape)

for i in range(nW):
    for j in range(nW):
        FCD_inter[i, j] = np.corrcoef(inter_vectors[i], inter_vectors[j])[0, 1]

FCD_inter_upper = FCD_inter[np.triu_indices_from(FCD_inter, k=1)]
norm_inter = np.sum(FCD_inter_upper ** 2)

# Final FCD dynamics difference
FCD_diff = norm_inter - norm_full

# =============================================================
# 4) FINAL METRIC: FCD_diff
# =============================================================

FCD_diff = norm_inter - norm_full

print("\n========= RESULTS =========")
print("FCD_VAR_OV_vect:", FCD_VAR_OV_vect)
print("homotopic_mean:", homotopic_mean)
print("Number of windows:", n_windows)
print("norm_full:", norm_full)
print("norm_inter:", norm_inter)
print("FCD_diff:", FCD_diff)
print("===========================")