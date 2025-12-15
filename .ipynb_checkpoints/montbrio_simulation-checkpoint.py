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

    # cut the initial transient (16s)
    bold_t = bold_t[8:]
    bold_d = bold_d[8:]
    FCD, _ = utils.compute_fcd(bold_d[:,0,:,0], win_len=5)
    FCD_VAR_OV_vect= np.var(np.triu(FCD, k=5))

    
    # =============================================================
    # compute FC from BOLD
    # =============================================================
    
    bold_ts = bold_d[:, 0, :, 0]   # shape (T, N)
    T, N = bold_ts.shape
    print("BOLD shape (T, N):", T, N)

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
    T = bold_ts.shape[0]
    N = bold_ts.shape[1]
    n_windows = (T - win_len) // step + 1

    FC_windows = []

    for w in range(n_windows):
        segment = bold_ts[w : w + win_len]
        FCw = np.corrcoef(segment.T)
        FC_windows.append(FCw)

    FC_windows = np.array(FC_windows)  # shape (W, N, N)

    # 2) Full-brain FCD upper triangle squared L2 norm
    FCD_full = FCD  # already computed above
    FCD_full_upper = FCD_full[np.triu_indices_from(FCD_full, k=1)]
    norm_full = np.sum(FCD_full_upper ** 2)

    # 3) Interhemispheric FCD (i <-> i+N/2)
    half = N // 2
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
    # Calculate time taken
    end_time = time.time()
    time_taken = end_time - start_time

    #save FCD_file
    FCD_file=FCD_file_pattern.format(subject=subject,noise=my_noise,G=G,dt=dt)
    np.save(FCD_file, FCD)

    return([FCD_VAR_OV_vect, homotopic_mean, FCD_diff, time_taken])
