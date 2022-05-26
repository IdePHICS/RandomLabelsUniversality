import csv
import h5py
import cvxpy as cp
import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def get_synthetic_data(N, M, which_cov, which_mean, path_to_data_folder, filename_p, filename_m):
    if which_cov == 'iid':
        if which_mean == "zero_mean":
            m = 0.
            X = np.random.normal(m, 1., size = (M,N))
    elif which_cov == 'gm_both_diagonal':
        cov = np.eye(N,N)
        if which_mean == "both_on_x_axis":
            m = np.zeros(N)
            m[0] = 1.
            y_mixture = np.random.choice([-1, 1], size=(M,1), p=[0.5, 0.5])
            mean = y_mixture @ np.reshape(m, (N,1)).T
        X = mean + np.random.multivariate_normal(np.zeros(N), cov, size = (M,))
    elif which_cov == 'gm_generic':
        y_mixture = np.random.choice([-1, 1], size=(M,1), p=[0.5, 0.5])
        filename = path_to_data_folder + filename_p
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_p = (X.T @ X)/M_tot
        filename = path_to_data_folder + filename_m
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_m = (X.T @ X)/M_tot

        samples_p = np.random.multivariate_normal(np.zeros(N), cov_p, size = (M,))
        samples_m = np.random.multivariate_normal(np.zeros(N), cov_m, size = (M,))
        X = []
        for μ in range(M):
            if which_mean == "both_on_x_axis":
                if y_mixture[μ] == 1:
                    X.append(np.concatenate([y_mixture[μ],np.zeros(N - 1)]) + samples_p[μ])
                else:
                    X.append(np.concatenate([y_mixture[μ], np.zeros(N - 1)]) + samples_m[μ])
    elif which_cov == 'gm_generic_same_trace':
        y_mixture = np.random.choice([-1, 1], size=(M,1), p=[0.5, 0.5])
        filename = path_to_data_folder + filename_p
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_p = (X.T @ X)/M_tot
        filename = path_to_data_folder + filename_m
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_m = (X.T @ X)/M_tot
            print(np.diag(cov_m))
            np.fill_diagonal(cov_p, np.diag(cov_m))
            print(np.diag(cov_p))

        samples_p = np.random.multivariate_normal(np.zeros(N), cov_p, size = (M,))
        samples_m = np.random.multivariate_normal(np.zeros(N), cov_m, size = (M,))
        X = []
        for μ in range(M):
            if which_mean == "both_on_x_axis":
                if y_mixture[μ] == 1:
                    X.append(np.concatenate([y_mixture[μ],np.zeros(N - 1)]) + samples_p[μ])
                else:
                    X.append(np.concatenate([y_mixture[μ], np.zeros(N - 1)]) + samples_m[μ])
    elif which_cov == 'gm_diagonal_two_blocks':
        y_mixture = np.random.choice([-1, 1], size=(M,1), p=[0.5, 0.5])
        cov_p = np.eye(N,N)
        diag = np.array([0.01 for i in range(0,N)])
        for i in range(int(N/2),N):
            diag[i] = 0.5
        np.fill_diagonal(cov_p,diag)

        cov_m = np.eye(N,N)
        diag = np.array([0.5 for i in range(0,N)])
        for i in range(int(N/2),N):
            diag[i] = 0.01
        np.fill_diagonal(cov_m,diag)

        samples_p = np.random.multivariate_normal(np.zeros(N), cov_p, size = (M,))
        samples_m = np.random.multivariate_normal(np.zeros(N), cov_m, size = (M,))
        X = []
        for μ in range(M):
            if which_mean == "both_on_x_axis":
                if y_mixture[μ] == 1:
                    X.append(np.concatenate([y_mixture[μ],np.zeros(N - 1)]) + samples_p[μ])
                else:
                    X.append(np.concatenate([y_mixture[μ], np.zeros(N - 1)]) + samples_m[μ])
    elif which_cov == 'gm_diagonal_three_blocks':
        y_mixture = np.random.choice([-1, 1], size=(M,1), p=[0.5, 0.5])
        cov_p = np.eye(N,N)
        diag = np.array([0.01 for i in range(0,N)])
        for i in range(int(N/3),int(N/3)*2):
            diag[i] = 0.98
        for i in range(int(N/3)*2,int(N/3)*3):
            diag[i] = 0.01
        np.fill_diagonal(cov_p,diag)

        cov_m = np.eye(N,N)
        diag = np.array([ 0.495 for i in range(0,N)])
        for i in range(int(N/3),int(N/3)*2):
            diag[i] = 0.01
        for i in range(int(N/3)*2,int(N/3)*3):
            diag[i] =  0.495
        np.fill_diagonal(cov_m,diag)

        samples_p = np.random.multivariate_normal(np.zeros(N), cov_p, size = (M,))
        samples_m = np.random.multivariate_normal(np.zeros(N), cov_m, size = (M,))
        X = []
        for μ in range(M):
            if which_mean == "both_on_x_axis":
                if y_mixture[μ] == 1:
                    X.append(np.zeros(N) + samples_p[μ])
                else:
                    X.append(np.zeros(N) + samples_m[μ])
    elif which_cov == 'sg_generic':
        if which_mean == "zero_mean":
            m = np.zeros(N)
        filename = path_to_data_folder + filename_p
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_p = (X.T @ X)/M_tot
        filename = path_to_data_folder + filename_m
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_m = (X.T @ X)/M_tot
        cov = 0.5 * cov_p + 0.5 * cov_m
        X = np.random.multivariate_normal(m, cov, size = (M,))
    elif which_cov == 'sg_generic_same_trace':
        if which_mean == "zero_mean":
            m = np.zeros(N)
        filename = path_to_data_folder + filename_p
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_p = (X.T @ X)/M_tot
        filename = path_to_data_folder + filename_m
        with h5py.File(filename, "r") as f:
            # List all groups
            print("Keys: %s" % f.keys())
            a_group_key = list(f.keys())[0]
            # Get the data
            X = np.asarray(list(f[a_group_key]))
            M_tot, N = X.shape
            cov_m = (X.T @ X)/M_tot
            np.fill_diagonal(cov_m, np.diag(cov_p))
        cov = 0.5 * cov_p + 0.5 * cov_m
        X = np.random.multivariate_normal(m, cov, size = (M,))
    elif which_cov == 'sg_diagonal_two_blocks':
        if which_mean == "zero_mean":
            m = np.zeros(N)
        cov_p = np.eye(N,N)
        diag = np.array([0.01 for i in range(0,N)])
        for i in range(int(N/2),N):
            diag[i] = 0.5
        np.fill_diagonal(cov_p,diag)

        cov_m = np.eye(N,N)
        diag = np.array([0.5 for i in range(0,N)])
        for i in range(int(N/2),N):
            diag[i] = 0.01
        np.fill_diagonal(cov_m,diag)
        np.fill_diagonal(cov_m,diag)
        cov = 0.5 * cov_p + 0.5 * cov_m
        X = np.random.multivariate_normal(m, cov, size = (M,))
    elif which_cov == 'sg_diagonal_three_blocks':
        if which_mean == "zero_mean":
            m = np.zeros(N)

        cov_p = np.eye(N,N)
        diag = np.array([0.01 for i in range(0,N)])
        for i in range(int(N/3),int(N/3)*2):
            diag[i] = 0.98
        for i in range(int(N/3)*2,int(N/3)*3):
            diag[i] = 0.01
        np.fill_diagonal(cov_p,diag)

        cov_m = np.eye(N,N)
        diag = np.array([0.495 for i in range(0,N)])
        for i in range(int(N/3),int(N/3)*2):
            diag[i] = 0.01
        for i in range(int(N/3)*2,int(N/3)*3):
            diag[i] = 0.495
        np.fill_diagonal(cov_m,diag)
        cov = 0.5 * cov_p + 0.5 * cov_m
        X = np.random.multivariate_normal(m, cov, size = (M,))
    y = np.random.choice([-1, 1], size=M, p=[0.5, 0.5])
    return np.array(X), y

def get_real_data(which_dataset, which_transform, non_lin):
    filename = "./data/X_%s_%s_%s.hdf5"%(which_dataset, which_transform, non_lin)
    with h5py.File(filename, "r") as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
        # Get the data
        X = np.asarray(list(f[a_group_key]))
    M, N = X.shape
    y = np.random.choice([-1, 1], size=M, p=[0.5, 0.5])
    return X, y

def ridge_estimator(X, y, lamb=0.1):
    '''
    Implements the pseudo-inverse ridge estimator.
    '''
    m, n = X.shape
    if m >= n:
        return np.linalg.inv(X.T @ X + lamb*np.identity(n)) @ X.T @ y
    elif m < n:
        return X.T @ np.linalg.inv(X @ X.T + lamb*np.identity(m)) @ y

def get_estimator(X_train, y_train, λ, loss = "square", solver = 'cvxpy'):
    M, N = X_train.shape
    if loss == "square":
        W = ridge_estimator(X_train/np.sqrt(N), y_train,lamb=λ)
    elif loss == 'logistic':
        W = LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, C = λ**(-1),
                               max_iter=1e4, tol=1e-5, verbose=0).fit(X_train/np.sqrt(N),y_train).coef_[0]
    elif loss == 'hinge':
        if solver == 'cvxpy':
            W = cp.Variable((N))
            l = cp.sum(cp.pos(1 - cp.multiply(y_train, (X_train @ W/np.sqrt(N)))))
            reg = (cp.norm(W, 2))**2
            lambd = cp.Parameter(nonneg=True)
            prob = cp.Problem(cp.Minimize(l + lambd*reg))
            lambd.value = λ
            prob.solve()
            W = W.value
        elif solver == 'sk':
            tol = 1e-5
            maxiter = 10000
            W = LinearSVC(penalty='l2', loss='hinge', dual=True, tol=tol, C=λ**(-1), multi_class='ovr',fit_intercept=False, intercept_scaling=0.0, class_weight=None,
                          verbose=False, random_state=None, max_iter=maxiter).fit(X_train/np.sqrt(N), y_train).coef_[0]
    return W

def mse(y,yhat):
    return np.mean((y-yhat)**2)
def logistic_loss(z):
    return np.log(1+np.exp(-z))
def hinge_loss(z):
    return np.maximum(np.zeros(len(z)), np.ones(len(z)) - z)

def get_train_loss(X_train, y_train, W, loss = "square"):
    M, N = X_train.shape
    if loss == "square":
        train_loss = 0.5*mse(y_train, (X_train @ W)/np.sqrt(N))
    elif loss == "logistic":
        train_loss = np.mean(logistic_loss((X_train @ W)/np.sqrt(N) * y_train))
    elif loss == "hinge":
        train_loss = np.mean(hinge_loss((X_train @ W)/np.sqrt(N) * y_train))
    return train_loss

def learning_curve(N = 1000, λ = 1e-15, data_type = 'real', loss = 'logistic', which_dataset = 'MNIST',
                   which_transform = 'wavelet_scattering', non_lin = 'erf', which_cov = 'both_diagonal',
                   which_mean = "both_on_x_axis", path_to_data_folder = './', filename_p = "X_p",
                   filename_m = "X_m", resfile = "res", solver = 'cvxpy', seed = 1):

    sim = {'N': [], 'alpha': [], 'data_type': [], 'which_dataset': [], 'which_transform': [], 'non_lin': [],
           'which_cov': [], 'which_mean': [], 'loss': [], 'lamb': [], 'train_loss': []}

    if data_type == 'real':
        X, y = get_real_data(which_dataset, which_transform, non_lin)

    αs =  np.linspace(0.05,5.0, 10)
    for α in αs:
        M = int(np.floor(α * N))

        if data_type == 'synthetic':
            X_train, y_train = get_synthetic_data(N, M, which_cov, which_mean)
        elif data_type == 'real':
            inds = np.random.choice(range(X.shape[0]), size=M, replace=False)
            X_train = X[inds, :]
            y_train = y[inds]
        M, N = X_train.shape
        W = get_estimator(X_train, y_train, λ, loss = loss, solver = solver)
        train_loss = get_train_loss(X_train, y_train, W, loss = loss)

        sim['N'].append(N)
        sim['alpha'].append(α)
        sim['data_type'].append(data_type)
        sim['which_dataset'].append(which_dataset)
        sim['which_transform'].append(which_transform)
        sim['non_lin'].append(non_lin)
        sim['which_cov'].append(which_cov)
        sim['which_mean'].append(which_mean)
        sim['loss'].append(loss)
        sim['lamb'].append(str(λ))
        sim['train_loss'].append(train_loss)

    sim = pd.DataFrame.from_dict(sim)
    if os.path.isfile(path_to_res_folder +  resfile) == False:
        with open(path_to_res_folder + "/seeds/" + resfile + "_seed_%d.csv"%(seed), mode='w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(sim.keys().to_list())
    sim.to_csv(path_to_res_folder + "/seeds/" + resfile + "_seed_%d.csv"%(seed), mode = 'a', index = False, header = None)
    return
