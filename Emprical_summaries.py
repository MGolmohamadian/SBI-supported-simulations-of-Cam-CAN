import sys, os, time
import numpy as np
import showcase1_ageing as utils
sys.path.append("/home/mgolmoha/TVB/virtual_ageing")
from showcase1_ageing.analysis_empirical import compute_fcd_filt_empirical, compute_fc_empirical, get_masks, compute_fcd_empirical

# --------------------------------------------------
# Load empirical BOLD data
# --------------------------------------------------
ts = np.loadtxt("ts_TVBSchaeferTian120.txt")   # shape: (T, 120)

# --------------------------------------------------
# Match simulation length (144 time points)
# --------------------------------------------------
ts = ts[8:30, :]   # EXACTLY match simulation length

T, N = ts.shape

NHALF = N // 2

# --------------------------------------------------
# Feature A: Homotopic Functional Connectivity
# --------------------------------------------------
rsFC = compute_fc_empirical(ts)
HOMO_FC = np.mean(np.diag(rsFC, k=NHALF))

# --------------------------------------------------
# Feature B: Functional Connectivity Dynamics (FCD)
# --------------------------------------------------
FCD, _ = compute_fcd_empirical(ts, win_len=5)

# Interhemispheric FCD
_, inter_mask = get_masks(N)
FCD_inter, fc_stack_inter, _ = compute_fcd_filt_empirical(ts, inter_mask, win_len=5)

# --------------------------------------------------
# Feature B: FCD variance (TWO methods)
# --------------------------------------------------

# B1 — Your original (statistical variance)
FCD_VAR_OV_vect = np.var(np.triu(FCD, k=5))
FCD_VAR_OV_INTER_vect = np.var(np.triu(FCD_inter, k=5))
FCD_DIFF_VAR = FCD_VAR_OV_INTER_vect - FCD_VAR_OV_vect

# B2 — Paper-accurate (Frobenius norm squared)
FCD_VAR_OV_FROB = np.sum(np.triu(FCD, k=5) ** 2)
FCD_VAR_OV_INTER_FROB = np.sum(np.triu(FCD_inter, k=5) ** 2)
FCD_DIFF_FROB = FCD_VAR_OV_INTER_FROB - FCD_VAR_OV_FROB


# --------------------------------------------------
# Feature C: Interhemispheric FC variability (TWO methods)
# --------------------------------------------------

# C1 — Your original (global std)
INTER_FC_STD_GLOBAL = np.std(fc_stack_inter)

# C2 — Paper-accurate (edge-wise temporal SD)
edgewise_std = np.std(fc_stack_inter, axis=0)
INTER_FC_STD_EDGEWISE = np.mean(edgewise_std)

# --------------------------------------------------
# Final feature vector (same order as simulation)
# --------------------------------------------------
features = [
    HOMO_FC,
    FCD_DIFF_FROB,
    INTER_FC_STD_EDGEWISE,
    FCD_DIFF_VAR,
    INTER_FC_STD_GLOBAL
]

print("Extracted features:")
print(features)

print("ts shape:", ts.shape)

# print("rsFC shape:", rsFC.shape)

# print("FCD shape:", FCD.shape)

# print("FCD_inter shape:", FCD_inter.shape)

# print("fc_stack_inter shape:", fc_stack_inter.shape)



with open("sub-CC320088_metrics.tsv", "w") as f:
    f.write(f"HOMO_FC\tFCD_DIFF_FROB\tINTER_FC_STD_EDGEWISE\tFCD_DIFF_VAR\tINTER_FC_STD_GLOBAL\n")
    f.write(f"{HOMO_FC}\t{FCD_DIFF_FROB}\t{INTER_FC_STD_EDGEWISE}\t{FCD_DIFF_VAR}\t{INTER_FC_STD_GLOBAL}\n")

print("Metrics saved to TSV file successfully!")