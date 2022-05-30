import csv
import h5py
import cvxpy as cp
import sys
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def get_synthetic_data(n, μ, Σ, μ_p, μ_m, Σp, Σm, which_synthetic_dataset):

    if which_synthetic_dataset == "single_gaussian":
        X = np.random.multivariate_normal(μ, Σ, size = (n,))
    elif which_synthetic_dataset == 'gaussian_mixture':
        p = len(μ_p)
        y_mixture = np.random.choice([-1, 1], size=(n,1), p=[0.5, 0.5])
        samples_p = np.random.multivariate_normal(np.zeros(p), Σp, size = (n,))
        samples_m = np.random.multivariate_normal(np.zeros(p), Σm, size = (n,))
        X = []
        for μ in range(n):
            if y_mixture[μ] == 1:
                X.append(μ_p + samples_p[μ])
            else:
                X.append(μ_m + samples_m[μ])
    y = np.random.choice([-1, 1], size=n, p=[0.5, 0.5])
    return np.array(X), y

def get_real_data(which_dataset, which_transform, which_non_lin):
    filename = "./data/X_%s_%s_%s.hdf5"%(which_dataset, which_transform, which_non_lin)
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

def get_estimator(X_train, y_train, λ, loss = "square_loss", solver = 'cvxpy'):
    n, p = X_train.shape
    if loss == "square_loss":
        W = ridge_estimator(X_train/np.sqrt(p), y_train,lamb=λ)
    elif loss == 'logistic_loss':
        W = LogisticRegression(penalty='l2',solver='lbfgs',fit_intercept=False, C = λ**(-1),
                               max_iter=1e4, tol=1e-5, verbose=0).fit(X_train/np.sqrt(p),y_train).coef_[0]
    elif loss == 'hinge_loss':
        if solver == 'cvxpy':
            W = cp.Variable((p))
            l = cp.sum(cp.pos(1 - cp.multiply(y_train, (X_train @ W/np.sqrt(p)))))
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
                          verbose=False, random_state=None, max_iter=maxiter).fit(X_train/np.sqrt(p), y_train).coef_[0]
    return W

def mse(y,yhat):
    return np.mean((y-yhat)**2)
def logistic_loss(z):
    return np.log(1+np.exp(-z))
def hinge_loss(z):
    return np.maximum(np.zeros(len(z)), np.ones(len(z)) - z)

def get_train_loss(X_train, y_train, W, loss = "square"):
    n, p = X_train.shape
    if loss == "square_loss":
        train_loss = 0.5*mse(y_train, (X_train @ W)/np.sqrt(p))
    elif loss == "logistic_loss":
        train_loss = np.mean(logistic_loss((X_train @ W)/np.sqrt(p) * y_train))
    elif loss == "hinge_loss":
        train_loss = np.mean(hinge_loss((X_train @ W)/np.sqrt(p) * y_train))
    return train_loss

def get_learning_curve_real(αs = [0.1], λ = 1e-15, loss = 'logistic', which_real_dataset = 'MNIST', which_transform = 'wavelet_scattering', which_non_lin = 'erf',
                        path_to_data_folder = './', path_to_res_folder = './', solver = 'cvxpy', seed = 1):

    sim = {'alpha': [], 'train_loss': [], 'p': [], 'which_real_dataset': [], 'which_transform': [], 'which_non_lin': [],
           'loss': [], 'lamb': []}

    X, y = get_real_data(which_real_dataset, which_transform, which_non_lin)
    n, p = X.shape

    for α in αs:
        print("-----------------------> processing n/p = ", α)
        n = int(np.floor(α * p))

        inds = np.random.choice(range(X.shape[0]), size=n, replace=False)
        X_train = X[inds, :]
        y_train = y[inds]
        n, p = X_train.shape
        W = get_estimator(X_train, y_train, λ, loss = loss, solver = solver)
        train_loss = get_train_loss(X_train, y_train, W, loss = loss)

        sim['p'].append(p)
        sim['alpha'].append(α)
        sim['which_real_dataset'].append(which_real_dataset)
        sim['which_transform'].append(which_transform)
        sim['which_non_lin'].append(which_non_lin)
        sim['loss'].append(loss)
        sim['lamb'].append(str(λ))
        sim['train_loss'].append(train_loss)

    if os.path.isdir(path_to_res_folder) == False: # create the results folder if not there
        try:
            os.makedirs(path_to_res_folder)
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    if os.path.isdir(path_to_res_folder + "/seeds") == False: # create the seed folder if not there
        try:
            os.makedirs(path_to_res_folder  + "/seeds")
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    sim = pd.DataFrame.from_dict(sim)
    if os.path.isfile(path_to_res_folder  + "/seeds" + "/sim_%s_seed_%d_real.csv"%(loss,seed)) == False:
        with open(path_to_res_folder + "/seeds" + "/sim_%s_seed_%d_real.csv"%(loss,seed), mode='w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(sim.keys().to_list())
    sim.to_csv(path_to_res_folder + "/seeds/" + "/sim_%s_seed_%d_real.csv"%(loss,seed), mode = 'a', index = False, header = None)
    return


def get_learning_curve_synthetic(p, μ, Σ, μ_p, Σp, μ_m, Σm, αs = [0.1], λ = 1e-15, loss = 'logistic', which_synthetic_dataset = 'single_gaussian',
                             mean_identifier = 'zero_mean', cov_identifier = 'both_diagonal', which_mean = "both_on_x_axis",
                             path_to_res_folder = './', solver = 'cvxpy', seed = 1):

    sim = {'alpha': [], 'train_loss': [], 'p': [], 'which_synthetic_dataset': [], 'mean_identifier': [], 'cov_identifier': [], 'loss': [], 'lamb': []}

    for α in αs:
        print("-----------------------> processing n/p = ", α)
        n = int(np.floor(α * p))

        X_train, y_train = get_synthetic_data(n, μ, Σ, μ_p, μ_m, Σp, Σm, which_synthetic_dataset)
        W = get_estimator(X_train, y_train, λ, loss = loss, solver = solver)
        train_loss = get_train_loss(X_train, y_train, W, loss = loss)

        sim['alpha'].append(α)
        sim['train_loss'].append(train_loss)
        sim['p'].append(p)
        sim['which_synthetic_dataset'].append(which_synthetic_dataset)
        sim['mean_identifier'].append(cov_identifier)
        sim['cov_identifier'].append(cov_identifier)
        sim['loss'].append(loss)
        sim['lamb'].append(str(λ))


    if os.path.isdir(path_to_res_folder) == False: # create the results folder if not there
        try:
            os.makedirs(path_to_res_folder)
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    if os.path.isdir(path_to_res_folder + "/seeds") == False: # create the seed folder if not there
        try:
            os.makedirs(path_to_res_folder  + "/seeds")
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    sim = pd.DataFrame.from_dict(sim)
    if os.path.isfile(path_to_res_folder + "/seeds" + "/sim_%s_%s_seed_%d.csv"%(loss,which_synthetic_dataset,seed)) == False:
        with open(path_to_res_folder + "/seeds" + "/sim_%s_%s_seed_%d.csv"%(loss,which_synthetic_dataset,seed), mode='w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(sim.keys().to_list())
    sim.to_csv(path_to_res_folder + "/seeds" + "/sim_%s_%s_seed_%d.csv"%(loss,which_synthetic_dataset,seed), mode = 'a', index = False, header = None)
    return
