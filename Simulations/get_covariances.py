import numpy as np
import os
import pandas as pd
from scipy import special
import matplotlib.pyplot as plt
import csv
import h5py
import cvxpy as cp
import torchvision.datasets as datasets
import torch
from torchvision import transforms
from kymatio.sklearn import Scattering2D

def build_covariances_synthetic_single_gaussian(p, which_cov = "none", which_mean = "none", ρ_p = 0.5, ρ_m = 0.5):

    if which_cov == "iid_Gaussian" and which_mean == "zero_mean":
        μ = np.zeros(p)
        Σ = np.eye(p,p)
    elif which_cov == "diagonal_three_blocks_sg" and which_mean == "zero_mean":
        μ = np.zeros(p)
        Σp = np.eye(p,p)
        diag = np.array([0.01 for i in range(0,p)])
        for i in range(int(p/3),int(p/3)*2):
            diag[i] = 0.98
        for i in range(int(p/3)*2,int(p/3)*3):
            diag[i] = 0.01
        np.fill_diagonal(Σp,diag)
        Σm = np.eye(p,p)
        diag = np.array([0.495 for i in range(0,p)])
        for i in range(int(p/3),int(p/3)*2):
            diag[i] = 0.01
        for i in range(int(p/3)*2,int(p/3)*3):
            diag[i] = 0.495
        np.fill_diagonal(Σm,diag)
        Σ = 0.5 * Σp + 0.5 * Σm
    return (μ, Σ)


def build_covariances_synthetic_gaussian_mixture(p, which_cov = "none", which_mean = "none", ρ_p = 0.5, ρ_m = 0.):

    if which_cov == "both_diagonal" and which_mean == "opposite_on_x_axis":
        μ_p = np.zeros(p)
        μ_p[0] = 1
        μ_m = np.zeros(p)
        μ_m[0] = -1
        Σp = np.eye(p,p)
        Σm = np.eye(p,p)
    elif which_cov == "diagonal_three_blocks_gm" and which_mean == "zero_mean":
        μ_p = np.zeros(p)
        μ_m = μ_p
        Σp = np.eye(p,p)
        diag = np.array([0.01 for i in range(0,p)])
        for i in range(int(p/3),int(p/3)*2):
            diag[i] = 0.98
        for i in range(int(p/3)*2,int(p/3)*3):
            diag[i] = 0.01
        np.fill_diagonal(Σp,diag)
        Σm = np.eye(p,p)
        diag = np.array([0.495 for i in range(0,p)])
        for i in range(int(p/3),int(p/3)*2):
            diag[i] = 0.01
        for i in range(int(p/3)*2,int(p/3)*3):
            diag[i] = 0.495
        np.fill_diagonal(Σm,diag)
    return (μ_p, μ_m, Σp, Σm)




def build_covariances_real(p, which_real_dataset = "none", which_transform = "none",
                    which_non_lin = "none", path_to_data_folder = "./"):

    μ = np.zeros(p)
    filename = path_to_data_folder + "/X_%s_%s_%s.hdf5"%(which_real_dataset, which_transform, which_non_lin)
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        X = np.asarray(list(f[a_group_key]))
        M_tot, N = X.shape
        Σ = (X.T @ X)/M_tot # compute the empirical covariance matrix
    return (μ, Σ)
