import csv
import h5py
import sys
import os
import numpy as np
import pandas as pd
from scipy import special
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

#### auxiliary functions for logistic regression ####

def gaussian(x, mean=0, var=1):
    """
    This function implements a Gaussian distribution.
    Input: x = Gaussian variable, type = float;
           mean = mean of the Gaussian distribution, type = float;
           var = variance of the Gaussian distribution, type = float.
    Output: Gaussian distribution evaluated in x, type = float.
    """
    return np.exp(-.5 * (x-mean)**2/var) / np.sqrt(2*np.pi*var)

def loss(z):
    """
    This function implements the logistic loss.
    Input: z = pre-activations time true label, type = float;
    Output: logistic loss evaluated in z, type = float.
    """
    return np.log(1 + np.exp(-z))

def moreau_loss(x, y, omega,V):
    """
    This function implements the Moreau envelope.
    Input: x = pre-activation, type = float;
           y = true label,type = float;
    Output: Moreau envelope evaluated in x and y, type = float.
    """
    return (x-omega)**2/(2*V) + loss(y*x)

def f_Vhat_plus(ξ, q, V):
    """
    This function implements the integrand of the saddle-point equation for the overlap q, relative to the case true label = 1.
    Input: ξ = auxiliary Gaussian variable of zero mean and unitary variance, type = float;
           q, V = overlap parameter, type = float.
    Output: integrand of the saddle-point equation for the overlap q, evaluated in ξ, q and V, type = float.
    """
    ω = np.sqrt(q)*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x'] # get the minimizer of the Moreau envelope
    return (1/(1/V + (1/4) * (1/np.cosh(λstar_plus/2)**2)))

def f_Vhat_minus(ξ, q, V):
    """
    This function implements the integrand of the saddle-point equation for the overlap q, relative to the case true label = -1.
    Input: ξ = auxiliary Gaussian variable of zero mean and unitary variance, type = float;
           q, V = overlap parameter, type = float.
    Output: integrand of the saddle-point equation for the overlap q, evaluated in ξ, q and V, type = float.
    """
    ω = np.sqrt(q)*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return (1/(1/V + (1/4) * (1/np.cosh(-λstar_minus/2)**2)))

def integrate_for_Vhat(q, V):
    """
    This function integrates the true label = 1 and true label = -1 integrand contributions to the saddle-point equation for the overlap q.
    Input: q, V = overlap parameter, type = float.
    Output: value of the integral, type = float.
    """
    I1 = quad(lambda ξ: f_Vhat_plus(ξ, q, V) * gaussian(ξ), -10, 10)[0]
    I2 = quad(lambda ξ: f_Vhat_minus(ξ, q, V) * gaussian(ξ), -10, 10)[0]
    return (1/2) * (I1 + I2)

def f_qhat_plus(ξ, q, V):
    """
    This function implements the integrand of the saddle-point equation for the overlap V, relative to the case true label = 1.
    Input: ξ = auxiliary Gaussian variable of zero mean and unitary variance, type = float;
           q, V = overlap parameter, type = float.
    Output: integrand of the saddle-point equation for the overlap q, evaluated in ξ, q and V, type = float.
    """
    ω = np.sqrt(q)*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    return (λstar_plus - ω)**2

def f_qhat_minus(ξ, q, V):
    """
    This function implements the integrand of the saddle-point equation for the overlap V, relative to the case true label = -1.
    Input: ξ = auxiliary Gaussian variable of zero mean and unitary variance, type = float;
           q, V = overlap parameter, type = float.
    Output: integrand of the saddle-point equation for the overlap q, evaluated in ξ, q and V, type = float.
    """
    ω = np.sqrt(q)*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    return (λstar_minus - ω)**2

def integrate_for_qhat(q, V):
    """
    This function integrates the true label = 1 and true label = -1 integrand contributions to the saddle-point equation for the overlap V.
    Input: q, V = overlap parameter, type = float.
    Output: value of the integral, type = float.
    """
    I1 = quad(lambda ξ: f_qhat_plus(ξ, q, V) * gaussian(ξ), -10, 10)[0]
    I2 = quad(lambda ξ: f_qhat_minus(ξ, q, V)* gaussian(ξ), -10, 10)[0]
    return (1/2) * (I1 + I2)

def integrand_training_error_plus_logistic(ξ, q, V):
    """
    This function implements the integrand to compute the true label = 1 contribution to the logistic training loss.
    Input:  ξ = auxiliary Gaussian variable of zero mean and unitary variance, type = float;
            q, V = overlap parameter, type = float.
    Output: value of the integrand, evaluated in ξ, q and V, type = float.
    """
    ω = np.sqrt(q)*ξ
    λstar_plus = minimize_scalar(lambda x: moreau_loss(x, 1, ω, V))['x']
    l_plus = loss(λstar_plus)
    return l_plus

def integrand_training_error_minus_logistic(ξ, q, V):
    """
    This function implements the integrand to compute the true label = -1 contribution to the logistic training loss.
    Input:  ξ = auxiliary Gaussian variable of zero mean and unitary variance, type = float;
            q, V = overlap parameter, type = float.
    Output: value of the integrand, evaluated in ξ, q and V, type = float.
    """
    ω = np.sqrt(q)*ξ
    λstar_minus = minimize_scalar(lambda x: moreau_loss(x, -1, ω, V))['x']
    l_minus = loss(-λstar_minus)
    return l_minus

def process_data(Σ):
    """
    This function computes the eigenvalues of a given matrix.
    Input: Σ = symmetric matrix, type = numpy array.
    Output: λ_Σ = eigenvalues, type = numpy array.
    """
    λ_Σ = np.linalg.eig(Σ)[0]
    return λ_Σ

#### Saddle-point equations ####

def dqh_Gs(qh, Vh, λ, λ_Σ):
    """
    This function implements the saddle-point equation for the overlap parameter V.
    Input: qh = conjugate overlap parameter, type = float;
           Vh = conjugate overlap parameter, type = float;
           λ = strength of the regularization, type = float;
           λ_Σ = eigenvalues of the input covariance matrix, type = numpy array.
    Output: partial derivative of the prior channel Gs as a function of the conjugate overlap parameter qh, type = float.
    """
    return (1/2) * np.mean(λ_Σ/ (λ + Vh * λ_Σ))

def dVh_Gs(qh, Vh, λ, λ_Σ):
    """
    This function implements the saddle-point equation for the overlap parameter q.
    Input: qh = conjugate overlap parameter, type = float;
           Vh = conjugate overlap parameter, type = float;
           λ = strength of the regularization, type = float;
           λ_Σ = eigenvalues of the input covariance matrix, type = numpy array.
    Output: partial derivative of the prior channel Gs as a function of the conjugate overlap parameter Vh, type = float.
    """
    return -(1/2) * np.mean((qh * λ_Σ) * λ_Σ / (λ + Vh * λ_Σ)**2)

def dq_Ge(q, V, loss = "square_loss"):
    """
    This function implements the saddle-point equation for the conjugate overlap parameter Vhat.
    Input: q = overlap parameter, type = float;
           V = overlap parameter, type = float;
           loss = which loss function to chose, type = string;
    Output: partial derivative of the prior channel Gs as a function of the overlap parameter q, type = float.
    """
    if loss == "square_loss":
        return -(1/2)*(1/(1 + V))
    elif loss == "hinge_loss":
        dq_Ge_p1 = -(2*q - (1-V)*V) * np.exp(-(1-V)**2/(2*q))/(4*np.sqrt(2*np.pi)*q**(3/2)) + (special.erf((1)/np.sqrt(2*q)) - special.erf((1-V)/np.sqrt(2*q)))/(4*V)
        dq_Ge_p2 = (2*q - (1-V)*V)/(4*np.sqrt(2*np.pi)*q**(3/2)) * np.exp(-(1-V)**2/(2*q))
        dq_Ge_m1 = -(2*q - (1-V)*V) * np.exp(-(1-V)**2/(2*q))/(4*np.sqrt(2*np.pi)*q**(3/2)) + (special.erf((1)/np.sqrt(2*q)) - special.erf((1-V)/np.sqrt(2*q)))/(4*V)
        dq_Ge_m2 = (2*q - (1-V)*V)/(4*np.sqrt(2*np.pi)*q**(3/2)) * np.exp(-(1-V)**2/(2*q))
        return -(1/2) * (dq_Ge_p1 + dq_Ge_p2 + dq_Ge_m1 + dq_Ge_m2)
    elif loss == "logistic_loss":
        Iq = integrate_for_qhat(q, V)
        res = -(1/2)*Iq/V**2
        return res

def dV_Ge(q, V, loss = "square_loss"):
    """
    This function implements the saddle-point equation for the conjugate overlap parameter q_hat.
    Input: q = overlap parameter, type = float;
           V = overlap parameter, type = float;
           loss = which loss function to chose, type = string;
    Output: partial derivative of the prior channel Gs as a function of the overlap parameter V, type = float.
    """
    if loss == "square_loss":
        return (1/2)*(1 + q)*(1 + V)**(-2)
    elif loss == "hinge_loss":
        dV_Ge_p1 = (V**3/np.sqrt(q) + np.sqrt(q)*(1+V)) * np.exp(-(1-V)**2/(2*q)) - np.sqrt(q)*np.exp(-1/(2*q))
        dV_Ge_p1 += -np.sqrt(np.pi/2) * ((1)**2 + q)*(special.erf(1/np.sqrt(2*q)) - special.erf((1-V)/np.sqrt(2*q)))
        dV_Ge_p1 /= 2*np.sqrt(2*np.pi)*V**2
        dV_Ge_p2 = - (V/(2*np.sqrt(2*np.pi*q)))*np.exp(-(1-V)**2/(2*q)) - (1/4)*(1 - special.erf(-(1-V)/np.sqrt(2*q)))
        dV_Ge_m1 = (V**3/np.sqrt(q) + np.sqrt(q)*(1+V)) * np.exp(-(1-V)**2/(2*q)) - np.sqrt(q)*np.exp(-1/(2*q))
        dV_Ge_m1 += -np.sqrt(np.pi/2) * (1 + q)*(special.erf(1/np.sqrt(2*q)) - special.erf((1-V)/np.sqrt(2*q)))
        dV_Ge_m1 /= 2*np.sqrt(2*np.pi)*V**2
        dV_Ge_m2 = - (V/(2*np.sqrt(2*np.pi*q)))*np.exp(-(1-V)**2/(2*q)) - (1/4)*(1 + special.erf((1-V)/np.sqrt(2*q)))
        return -(1/2) * (dV_Ge_p1 + dV_Ge_p2 + dV_Ge_m1 + dV_Ge_m2)
    elif loss == "logistic_loss":
        Iv = integrate_for_Vhat(q, V)
        res = (1/2)*((1/V) - (1/V**2) * Iv)
        return res

def train_loss(q, V, loss = "square_loss"):
    """
    This function computes the training loss for a given value of the overlap parameter q and V.
    Input: q,V = overlap parameter,  type = float;
           loss = which loss function to chose, type = string.
    Output: training loss, type = float.
    """
    if loss == "square_loss":
        return (1/2) * (1 + q)/(1 + V)**2
    if loss == "hinge_loss":
        res = np.sqrt(q/(2*np.pi))*np.exp(-(1-V)**2/(2*q)) + (1/2)*(1-V)*(1+special.erf((1-V)/np.sqrt(2*q)))
        res += np.sqrt(q/(2*np.pi))*np.exp(-(1-V)**2/(2*q)) + (1/2)*(1-V)*(1+special.erf((1-V)/np.sqrt(2*q)))
        return (1/2)*res
    if loss == "logistic_loss":
        res = quad(lambda ξ: integrand_training_error_plus_logistic(ξ, q, V) * gaussian(ξ), -10, 10)[0]
        res += quad(lambda ξ: integrand_training_error_minus_logistic(ξ, q, V) * gaussian(ξ), -10, 10)[0]
        return (1/2)*res

def iterate_saddle_point_equations(q_init, V_init, λ_Σ, α = 0.1, λ = 0.01, max_iters = 1000, ϵ = 1e-7, ψ = 0.0, loss = "square_loss"):
    """
    This function implements the saddle-point iterations.
    Input: q_init, V_init = initial value of the overlap parameter, type = float;
           λ_Σ = eigenvalues of the input covariance matrix, type = numpy array;
           α = number of samples per input dimension, type = float;
           λ = strength of the regularization, type = float;
           max_iters = maximum number of saddle-point iterations, type = int;
           ϵ = tollerance of convergence, type = float;
           ψ = dumping strength, type = float;
           loss = which loss function to chose, type = string.
    Output: q, V = overlap parameter at convergence, type = float;
            ok = flag assessing whether the saddle-point equations have converged or not, type = string.
    """
    ok = "false"
    for it in range(max_iters):
        qh = 2 * α * dV_Ge(q_init, V_init, loss = loss)
        Vh = -2 * α * dq_Ge(q_init, V_init, loss = loss)
        qh = qh * (1-ψ) + qh * ψ # if ψ = 0 -> no dumping
        Vh = Vh * (1-ψ) + Vh * ψ
        q = -2 * dVh_Gs(qh, Vh, λ, λ_Σ)
        V = 2 * dqh_Gs(qh, Vh, λ, λ_Σ)
        q = q * (1-ψ) + q_init * ψ
        V = V * (1-ψ) + V_init * ψ
        Δ = (np.abs(q - q_init) + np.abs(V - V_init))/2
        print("it = ", it, "Δ = ", Δ)
        if Δ < ϵ and it > 5:
            ok = "true"
            break
        q_init = q
        V_init = V
    return q, V, ok

############ Learning curve ############

def get_single_gaussian_learning_curve(Σ, mean_identifier, cov_identifier, αs=[0.1], λ=0.01, loss = "square_loss", path_to_res_folder = "./",
                                       resfile = "res.csv", max_iters = 1000, ϵ = 1e-7, ψ=0.):

    """
    This function implements the learning curve, e.g. the training loss as a function of the number of samples per input dimension (n/p).
    Input: p = input dimension, type = int;
           αs = set of possible values for n/p, type = numpy array;
           λ = regularization strength;
           data_type = type of dataset, accepted types are either "synthetic" or "real", type = string;
           which_cov = the covariance type to compute in case of synthetic data, type = string;
           ρ_p, ρ_m = probabilities of an input data-point to belong to the corresponding Gaussian cluster, type = int;
           loss = which loss function to chose, type = string;
           path_to_res_folder = path to the folder where to store the results;
           resfile = name of the file where to store the results in csv format;
           max_iters = maximum number of saddle-point iterations, type = int;
           ϵ = tollerance of convergence, type = float;
           ψ = dumping strength, type = float.
    Output: csv file reporting the training loss as a function of n/p.
    """
    p = len(Σ)
    res = {'alpha': [], 'train_loss': [], 'p': [], 'mean_identifier': [], 'cov_identifier':[], 'loss': [], 'lamb': [], 'q': [], 'V': [] , 'ok':[]}

    print("computing its eigenvalues...")
    λ_Σ = process_data(Σ)
    q_init = 0.2; V_init = 100.
    tl = 0.
    for α in αs:
        print("-----------------------> processing n/p = ", α)
        q, V, ok = iterate_saddle_point_equations(q_init, V_init, λ_Σ, α = α, λ = λ, max_iters = max_iters, ϵ = ϵ,
                                                  ψ = ψ, loss = loss)
        tl = train_loss(q, V, loss = loss)
        res["alpha"].append(α)
        res["train_loss"].append(tl)
        res["p"].append(p)
        res["mean_identifier"].append(cov_identifier)
        res["cov_identifier"].append(cov_identifier)
        res["loss"].append(loss)
        res["lamb"].append(λ)
        res["q"].append(q)
        res["V"].append(V)
        res["ok"].append(ok)
        q_init = q #start from the last convergence to speed-up the saddle-point iterations
        V_init = V

    print("collecting the results in the csv file...")
    if os.path.isdir(path_to_res_folder) == False: # create the results folder if not there
        try:
            os.makedirs(path_to_res_folder)
        except OSError:
            print ("\n!!! ERROR: Creation of the directory %s failed" % path_to_res_folder)
            raise

    res = pd.DataFrame.from_dict(res)
    if os.path.isfile(path_to_res_folder +  "/" + resfile) == False:
        with open(path_to_res_folder + "/" + resfile, mode='w') as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow(res.keys().to_list())
    res.to_csv(path_to_res_folder + "/" + resfile, mode = 'a', index = False, header = None)
    print("DONE!")
    return
