import sys, os, time
import numpy as np
import showcase1_ageing as utils
sys.path.append("/home/mgolmoha/TVB/virtual_ageing")
from showcase1_ageing.analysis import compute_fcd_filt, compute_fc, get_masks, compute_fcd
from tvb.simulator.lab import *
from tvb.simulator.backend.nb_mpr import NbMPRBackend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import resource
import time
import fcntl
from scipy.stats import ks_2samp


def get_connectivity(scaling_factor,weights_file):
        SC = np.loadtxt(weights_file)
        SC = SC / scaling_factor
        conn = connectivity.Connectivity(
                weights = SC,
                tract_lengths=np.ones_like(SC),
                centres = np.zeros(np.shape(SC)[0]),
                speed = np.r_[np.Inf]
        )
        conn.compute_region_labels()
        return conn


def process_sub(subject, my_noise, G, dt, sim_len, weights_file_pattern, FCD_file_pattern):
    start_time = time.time()

    weights_file=weights_file_pattern.format(subject=subject)
    

    dt      = dt
    nsigma  = my_noise
    G       = G
    sim_len = sim_len
    #30e3 produced 300000 ms, or 300s or 5 min. we get 142 bold timepoints if we have decimate 2000 and remove first eight timepoints

    sim = simulator.Simulator(
        connectivity = get_connectivity(1,weights_file),
        model = models.MontbrioPazoRoxin(
            eta   = np.r_[-4.6],
            J     = np.r_[14.5],
            Delta = np.r_[0.7],
            tau   = np.r_[1.],
        ),
        coupling = coupling.Linear(a=np.r_[G]),
        integrator = integrators.HeunStochastic(
            dt = dt,
            noise = noise.Additive(nsig=np.r_[nsigma, nsigma*2])
        ),
        monitors = [monitors.TemporalAverage(period=0.1)]
    ).configure()

    runner = NbMPRBackend()
    (tavg_t, tavg_d), = runner.run_sim(sim, simulation_length=sim_len)
    tavg_t *= 10
        
    bold_t, bold_d = utils.tavg_to_bold(tavg_t, tavg_d, tavg_period=1.,decimate=2000)
    print('tavg_to_bold step')

    # --------------------------------------------------
    # Remove transient
    # --------------------------------------------------
    bold_t = bold_t[8:]
    bold_d = bold_d[8:]

    ts = bold_d[:, 0, :, 0]     # [time, nodes]
    n_samples, n_nodes = ts.shape
    N = n_nodes
    NHALF = N // 2

    # --------------------------------------------------
    # Feature A: Homotopic FC (paper)
    # --------------------------------------------------
    rsFC=compute_fc(bold_d)
    HOMO_FC = np.mean(np.diag(rsFC, k=NHALF))
 
    # --------------------------------------------------
    # Full FCD
    # --------------------------------------------------
    FCD, _ = compute_fcd(ts, win_len=5)

    # --------------------------------------------------
    # Interhemispheric FCD
    # --------------------------------------------------
    _, inter_mask = get_masks(N)
    FCD_inter, fc_stack_inter, _ = compute_fcd_filt(ts, inter_mask, win_len=5)

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
    # Save FCD
    # --------------------------------------------------
    #FCD_file = FCD_file_pattern.format(
    #    subject=subject, noise=my_noise, G=G, dt=dt)
    #np.save(FCD_file, FCD)

    # --------------------------------------------------
    # Time
    # --------------------------------------------------
   # time_taken = time.time() - start_time

    # --------------------------------------------------
    # Return ALL outputs
    # -------------------------------------------------


    # FCD matrices
#   ts_shape=ts.shape
    rsFC_shape= rsFC.shape
    FCD_shape=FCD.shape
    FCD_inter_shape=FCD_inter.shape

    return([HOMO_FC, FCD_DIFF_FROB, INTER_FC_STD_EDGEWISE, FCD_DIFF_VAR, INTER_FC_STD_GLOBAL, FCD_shape[0], FCD_shape[1], FCD_inter_shape[0], FCD_inter_shape[1],  rsFC_shape[0],  rsFC_shape[1], NHALF])

