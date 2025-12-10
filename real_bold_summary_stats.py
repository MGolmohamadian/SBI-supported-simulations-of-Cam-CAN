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


bold_t = np.loadtxt("ts_TVBSchaeferTian120.txt")
print(bold_t.shape)
# ---------- 1) Compute FCD ----------
T, N = bold_t.shape

   # Cut the initial transient (16s) – remove first 8 time points
bold_ts = bold_t[176:, :]   # shape (T_new, N)   # shape (T_new, N)

# Compute FCD using utils.compute_fcd
# For 2D BOLD, we can pass bold_ts directly
FCD, _ = utils.compute_fcd(bold_ts, win_len=5)
FCD_VAR_OV_vect = np.var(np.triu(FCD, k=5))


# =============================================================
# compute FC from BOLD
# =============================================================

# FC matrix
FC = np.corrcoef(bold_ts.T)

# homotopic FC: pair region i with i+N/2
half = N // 2
homotopic_vals = FC[np.arange(half), np.arange(half) + half]
homotopic_mean = np.mean(homotopic_vals)
    

# =============================================================
# Compute FCD dynamics difference (σ_diff^2)
# =============================================================

# 1) We need sliding-window FC matrices
# utils.compute_fcd already computed the FCD matrix but not the FC windows.
# So compute them here manually.


win_len = 5  # same as in FCD computation
step = 1
T_new, N = bold_ts.shape
n_windows = (T_new - win_len) // step + 1

FC_windows = []

for w in range(n_windows):
    segment = bold_ts[w : w + win_len, :]
    FCw = np.corrcoef(segment.T)
    FC_windows.append(FCw)

FC_windows = np.array(FC_windows)  # shape (W, N, N)

# 2) Full-brain FCD upper triangle squared L2 norm
FCD_full_upper = FCD[np.triu_indices_from(FCD, k=1)]
norm_full = np.sum(FCD_full_upper ** 2)

# 3) Interhemispheric FCD (i <-> i+N/2)
lh = np.arange(half)
rh = lh + half

# extract interhemispheric edges for each window
inter_vectors = np.array([FCw[lh, rh] for FCw in FC_windows])

# compute interhemispheric FCD
nW = inter_vectors.shape[0]
FCD_inter = np.zeros((nW, nW))

for i in range(nW):
    for j in range(nW):
        FCD_inter[i, j] = np.corrcoef(inter_vectors[i], inter_vectors[j])[0, 1]

FCD_inter_upper = FCD_inter[np.triu_indices_from(FCD_inter, k=1)]
norm_inter = np.sum(FCD_inter_upper ** 2)

# Final FCD dynamics difference
FCD_diff = norm_inter - norm_full

# =============================================================
# Print results
print("FCD_VAR_OV_vect:", FCD_VAR_OV_vect)
print("homotopic_mean:", homotopic_mean)
print("FCD_diff:", FCD_diff)
print("---------- DONE ----------")

