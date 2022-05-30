import csv
import h5py
import sys
import os
import numpy as np
import pandas as pd
from scipy import special

#### Free-energy and saddle-point equations ####

def dmh_p_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m):
    p = len(μ_p)
    A = λ*np.eye(p,p) + Vh_p * Σp + Vh_m * Σm
    res = (1/p) * np.sqrt(p)*(mh_p * μ_p + mh_m * μ_m).T @ np.linalg.matrix_power(A,-1) * np.sqrt(p) @ μ_p
    return res

def dmh_m_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m):
    p = len(μ_m)
    A = λ*np.eye(p,p) + Vh_p * Σp + Vh_m * Σm
    res = (1/p) * np.sqrt(p)*(mh_p * μ_p + mh_m * μ_m).T @ np.linalg.matrix_power(A, -1) * np.sqrt(p) @ μ_m
    return res

def dqh_p_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m):
    p = len(μ_p)
    A = λ*np.eye(p,p) + Vh_p * Σp + Vh_m * Σm
    res = (1/2) * (1/p) * np.trace(Σp @ np.linalg.matrix_power(A, -1))
    return res

def dqh_m_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m):
    p = len(μ_m)
    A = λ*np.eye(p,p) + Vh_p * Σp + Vh_m * Σm
    res = (1/2) * (1/p) * np.trace(Σm @ np.linalg.matrix_power(A, -1))
    return res

def dVh_p_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m):
    p = len(μ_p)
    A = λ*np.eye(p,p) + Vh_p * Σp + Vh_m * Σm
    res = (1/p) * np.sqrt(p)*(mh_p * μ_p + mh_m * μ_m).T @ np.linalg.matrix_power(A,-2) @ Σp * np.sqrt(p) @ (mh_p * μ_p + mh_m * μ_m)
    res += (1/p) * np.trace((qh_p * Σp + qh_m * Σm) @ Σp @ np.linalg.matrix_power(A,-2))
    return -(1/2) * res

def dVh_m_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m):
    p = len(μ_m)
    A = λ*np.eye(p,p) + Vh_p * Σp + Vh_m * Σm
    res = (1/p) * np.sqrt(p)*(mh_p * μ_p + mh_m * μ_m).T @ np.linalg.matrix_power(A,-2) @ Σm * np.sqrt(p) @ (mh_p * μ_p + mh_m * μ_m)
    res += (1/p) * np.trace((qh_p * Σp + qh_m * Σm) @ Σm @ np.linalg.matrix_power(A,-2))
    return -(1/2) * res

def dm_p_Ge(m_p, m_m, q_p, q_m, V_p, V_m, loss = "square_loss"):
    if loss == "square_loss":
        res = m_p/(1 + V_p)
    elif loss == "hinge_loss":
        dm_p_Ge_p1 = - (q_p/(np.sqrt(2*np.pi*q_p)*V_p))*np.exp(-(1 - m_p)**2/(2*q_p))
        dm_p_Ge_p1 += ((2*q_p + V_p**2)/(2*V_p*np.sqrt(2*np.pi*q_p))) * np.exp(-((1-m_p)*(1 - m_p - 2*V_p) + V_p**2)/(2*q_p))
        dm_p_Ge_p1 += ((1 - m_p)/(2*V_p))*(special.erf((1-m_p-V_p)/np.sqrt(2*q_p)) - special.erf((1-m_p)/np.sqrt(2*q_p)))
        dm_p_Ge_p2 = -(V_p/(2*np.sqrt(2*np.pi*q_p))) * np.exp(-(1-m_p-V_p)**2/(2*q_p)) - (1/2)*(1+special.erf((1-m_p-V_p)/np.sqrt(2*q_p)))
        dm_p_Ge_m1 = - (q_p/(np.sqrt(2*np.pi*q_p)*V_p))*np.exp(-(1 + m_p)**2/(2*q_p))
        dm_p_Ge_m1 += -((2*q_p + V_p**2)/(2*V_p*np.sqrt(2*np.pi*q_p))) * np.exp(-((1+m_p)*(1 + m_p - 2*V_p) + V_p**2)/(2*q_p))
        dm_p_Ge_m1 += ((1 + m_p)/(2*V_p))*(special.erf((1+m_p)/np.sqrt(2*q_p))-special.erf((1+m_p-V_p)/np.sqrt(2*q_p)))
        dm_p_Ge_m2 = (V_p/(2*np.sqrt(2*np.pi*q_p))) * np.exp(-(1+m_p-V_p)**2/(2*q_p)) + (1/2)*(1+special.erf((1+m_p-V_p)/np.sqrt(2*q_p)))
        res = -(1/2) * (dm_p_Ge_p1 + dm_p_Ge_p2 + dm_p_Ge_m1 + dm_p_Ge_m2)
    return res

def dm_m_Ge(m_p, m_m, q_p, q_m, V_p, V_m, loss = "square_loss"):
    if loss == "square_loss":
        res =  m_m/(1 + V_m)
    elif loss == "hinge_loss":
        dm_m_Ge_p1 = - (q_m/(np.sqrt(2*np.pi*q_m)*V_m))*np.exp(-(1 - m_m)**2/(2*q_m))
        dm_m_Ge_p1 += ((2*q_m + V_m**2)/(2*V_m*np.sqrt(2*np.pi*q_m))) * np.exp(-((1-m_m)*(1 - m_m - 2*V_m) + V_m**2)/(2*q_m))
        dm_m_Ge_p1 += ((1 - m_m)/(2*V_m))*(special.erf((1-m_m-V_m)/np.sqrt(2*q_m)) - special.erf((1-m_m)/np.sqrt(2*q_m)))
        dm_m_Ge_p2 = -(V_m/(2*np.sqrt(2*np.pi*q_m))) * np.exp(-(1-m_m-V_m)**2/(2*q_m)) - (1/2)*(1+special.erf((1-m_m-V_m)/np.sqrt(2*q_m)))
        dm_m_Ge_m1 = - (q_m/(np.sqrt(2*np.pi*q_m)*V_m))*np.exp(-(1 + m_m)**2/(2*q_m))
        dm_m_Ge_m1 += -((2*q_m + V_m**2)/(2*V_m*np.sqrt(2*np.pi*q_m))) * np.exp(-((1+m_m)*(1 + m_m - 2*V_m) + V_m**2)/(2*q_m))
        dm_m_Ge_m1 += ((1 + m_m)/(2*V_m))*(special.erf((1+m_m)/np.sqrt(2*q_m))-special.erf((1+m_m-V_m)/np.sqrt(2*q_m)))
        dm_m_Ge_m2 = (V_m/(2*np.sqrt(2*np.pi*q_m))) * np.exp(-(1+m_m-V_m)**2/(2*q_m)) + (1/2)*(1+special.erf((1+m_m-V_m)/np.sqrt(2*q_m)))
        res = -(1/2) * (dm_m_Ge_p1 + dm_m_Ge_p2 + dm_m_Ge_m1 + dm_m_Ge_m2)
    return res

def dq_p_Ge(m_p, m_m, q_p, q_m, V_p, V_m, loss = "square_loss"):
    if loss == "square_loss":
        res =  -(1/2)*(1/(1 + V_p))
    elif loss == "hinge_loss":
        dq_p_Ge_p1 = -(2*q_p - (1-m_p-V_p)*V_p) * np.exp(-(1-m_p-V_p)**2/(2*q_p))/(4*np.sqrt(2*np.pi)*q_p**(3/2)) + (special.erf((1-m_p)/np.sqrt(2*q_p)) - special.erf((1-m_p-V_p)/np.sqrt(2*q_p)))/(4*V_p)
        dq_p_Ge_p2 = (2*q_p - (1-m_p-V_p)*V_p)/(4*np.sqrt(2*np.pi)*q_p**(3/2)) * np.exp(-(1-m_p-V_p)**2/(2*q_p))
        dq_p_Ge_m1 = -(2*q_p - (1+m_p-V_p)*V_p) * np.exp(-(1+m_p-V_p)**2/(2*q_p))/(4*np.sqrt(2*np.pi)*q_p**(3/2)) + (special.erf((1+m_p)/np.sqrt(2*q_p)) - special.erf((1+m_p-V_p)/np.sqrt(2*q_p)))/(4*V_p)
        dq_p_Ge_m2 = (2*q_p - (1+m_p-V_p)*V_p)/(4*np.sqrt(2*np.pi)*q_p**(3/2)) * np.exp(-(1+m_p-V_p)**2/(2*q_p))
        res =  -(1/2) * (dq_p_Ge_p1 + dq_p_Ge_p2 + dq_p_Ge_m1 + dq_p_Ge_m2)
    return res

def dq_m_Ge(m_p, m_m, q_p, q_m, V_p, V_m, loss = "square_loss"):
    if loss == "square_loss":
        res =  -(1/2)*(1/(1 + V_m))
    elif loss == "hinge_loss":
        dq_m_Ge_p1 = -(2*q_m - (1-m_m-V_m)*V_m) * np.exp(-(1-m_m-V_m)**2/(2*q_m))/(4*np.sqrt(2*np.pi)*q_m**(3/2)) + (special.erf((1-m_m)/np.sqrt(2*q_m)) - special.erf((1-m_m-V_m)/np.sqrt(2*q_m)))/(4*V_m)
        dq_m_Ge_p2 = (2*q_m - (1-m_m-V_m)*V_m)/(4*np.sqrt(2*np.pi)*q_m**(3/2)) * np.exp(-(1-m_m-V_m)**2/(2*q_m))
        dq_m_Ge_m1 = -(2*q_m - (1+m_m-V_m)*V_m) * np.exp(-(1+m_m-V_m)**2/(2*q_m))/(4*np.sqrt(2*np.pi)*q_m**(3/2)) + (special.erf((1+m_m)/np.sqrt(2*q_m)) - special.erf((1+m_m-V_m)/np.sqrt(2*q_m)))/(4*V_m)
        dq_m_Ge_m2 = (2*q_m - (1+m_m-V_m)*V_m)/(4*np.sqrt(2*np.pi)*q_m**(3/2)) * np.exp(-(1+m_m-V_m)**2/(2*q_m))
        res = -(1/2) * (dq_m_Ge_p1 + dq_m_Ge_p2 + dq_m_Ge_m1 + dq_m_Ge_m2)
    return res

def dV_p_Ge(m_p, m_m, q_p, q_m, V_p, V_m, loss = "square_loss"):
    if loss == "square_loss":
        res = (1/2)*(1 + m_p**2 + q_p)*(1 + V_p)**(-2)
    elif loss == "hinge_loss":
        dV_p_Ge_p1 = (V_p**3/np.sqrt(q_p) + np.sqrt(q_p)*(1-m_p+V_p)) * np.exp(-(1-m_p-V_p)**2/(2*q_p)) - (1-m_p)*np.sqrt(q_p)*np.exp(-(1-m_p)**2/(2*q_p))
        dV_p_Ge_p1 += -np.sqrt(np.pi/2) * ((1-m_p)**2 + q_p)*(special.erf((1-m_p)/np.sqrt(2*q_p)) - special.erf((1-m_p-V_p)/np.sqrt(2*q_p)))
        dV_p_Ge_p1 /= 2*np.sqrt(2*np.pi)*V_p**2
        dV_p_Ge_p2 = - (V_p/(2*np.sqrt(2*np.pi*q_p)))*np.exp(-(1-m_p-V_p)**2/(2*q_p)) - (1/4)*(1 - special.erf(-(1-m_p-V_p)/np.sqrt(2*q_p)))
        dV_p_Ge_m1 = (V_p**3/np.sqrt(q_p) + np.sqrt(q_p)*(1+m_p+V_p)) * np.exp(-(1+m_p-V_p)**2/(2*q_p)) - (1+m_p)*np.sqrt(q_p)*np.exp(-(1+m_p)**2/(2*q_p))
        dV_p_Ge_m1 += -np.sqrt(np.pi/2) * ((1+m_p)**2 + q_p)*(special.erf((1+m_p)/np.sqrt(2*q_p)) - special.erf((1+m_p-V_p)/np.sqrt(2*q_p)))
        dV_p_Ge_m1 /= 2*np.sqrt(2*np.pi)*V_p**2
        dV_p_Ge_m2 = - (V_p/(2*np.sqrt(2*np.pi*q_p)))*np.exp(-(1+m_p-V_p)**2/(2*q_p)) - (1/4)*(1 + special.erf((1+m_p-V_p)/np.sqrt(2*q_p)))
        res = -(1/2) * (dV_p_Ge_p1 + dV_p_Ge_p2 + dV_p_Ge_m1 + dV_p_Ge_m2)
    return res

def dV_m_Ge(m_p, m_m, q_p, q_m, V_p, V_m, loss = "square_loss"):
    if loss == "square_loss":
        res = (1/2)*(1 + m_m**2 + q_m)*(1 + V_m)**(-2)
    elif loss == "hinge_loss":
        dV_m_Ge_p1 = (V_m**3/np.sqrt(q_m) + np.sqrt(q_m)*(1-m_m+V_m)) * np.exp(-(1-m_m-V_m)**2/(2*q_m)) - (1-m_m)*np.sqrt(q_m)*np.exp(-(1-m_m)**2/(2*q_m))
        dV_m_Ge_p1 += -np.sqrt(np.pi/2) * ((1-m_m)**2 + q_m)*(special.erf((1-m_m)/np.sqrt(2*q_m)) - special.erf((1-m_m-V_m)/np.sqrt(2*q_m)))
        dV_m_Ge_p1 /= 2*np.sqrt(2*np.pi)*V_m**2
        dV_m_Ge_p2 = - (V_m/(2*np.sqrt(2*np.pi*q_m)))*np.exp(-(1-m_m-V_m)**2/(2*q_m)) - (1/4)*(1 - special.erf(-(1-m_m-V_m)/np.sqrt(2*q_m)))
        dV_m_Ge_m1 = (V_m**3/np.sqrt(q_m) + np.sqrt(q_m)*(1+m_m+V_m)) * np.exp(-(1+m_m-V_m)**2/(2*q_m)) - (1+m_m)*np.sqrt(q_m)*np.exp(-(1+m_m)**2/(2*q_m))
        dV_m_Ge_m1 += -np.sqrt(np.pi/2) * ((1+m_m)**2 + q_m)*(special.erf((1+m_m)/np.sqrt(2*q_m)) - special.erf((1+m_m-V_m)/np.sqrt(2*q_m)))
        dV_m_Ge_m1 /= 2*np.sqrt(2*np.pi)*V_m**2
        dV_m_Ge_m2 = - (V_m/(2*np.sqrt(2*np.pi*q_m)))*np.exp(-(1+m_m-V_m)**2/(2*q_m)) - (1/4)*(1 + special.erf((1+m_m-V_m)/np.sqrt(2*q_m)))
        res =  -(1/2) * (dV_m_Ge_p1 + dV_m_Ge_p2 + dV_m_Ge_m1 + dV_m_Ge_m2)
    return res

def train_loss(m_p, m_m, q_p, q_m, V_p, V_m, ρ_p, ρ_m, loss = "square_loss"):
    if loss == "square_loss":
        res = (ρ_p/2) * (1 + m_p**2 + q_p)/(1 + V_p)**2 + (ρ_m/2) * (1 + m_m**2 + q_m)/(1 + V_m)**2
    elif loss == "hinge_loss":
        plus = np.sqrt(q_p/(2*np.pi))*np.exp(-(1-m_p-V_p)**2/(2*q_p)) + (1/2)*(1-m_p-V_p)*(1+special.erf((1-m_p-V_p)/np.sqrt(2*q_p)))
        plus += np.sqrt(q_p/(2*np.pi))*np.exp(-(1+m_p-V_p)**2/(2*q_p)) + (1/2)*(1+m_p-V_p)*(1+special.erf((1+m_p-V_p)/np.sqrt(2*q_p)))
        minus = np.sqrt(q_m/(2*np.pi))*np.exp(-(1-m_m-V_m)**2/(2*q_m)) + (1/2)*(1-m_m-V_m)*(1+special.erf((1-m_m-V_m)/np.sqrt(2*q_m)))
        minus += np.sqrt(q_m/(2*np.pi))*np.exp(-(1+m_m-V_m)**2/(2*q_m)) + (1/2)*(1+m_m-V_m)*(1+special.erf((1+m_m-V_m)/np.sqrt(2*q_m)))
        return (ρ_p/2)*plus + (ρ_m/2)*minus
    return res

def iterate_saddle_point_equations(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, Σp, Σm, μ_p, μ_m, ρ_p =0.5, ρ_m = 0.5, α = 0.1, λ = 0.1, max_iters = 1000, ϵ = 1e-7, ψ = 0.0, loss = "square_loss"):
    ok = "false"
    for it in range(max_iters):
        mh_p = α * ρ_p * dm_p_Ge(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, loss = loss)
        mh_m = α * ρ_m * dm_m_Ge(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, loss = loss)
        Vh_p = -2 * α * ρ_p * dq_p_Ge(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, loss = loss)
        Vh_m = -2 * α * ρ_m * dq_m_Ge(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, loss = loss)
        qh_p = 2 * α * ρ_p *dV_p_Ge(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, loss = loss)
        qh_m = 2 * α * ρ_m * dV_m_Ge(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, loss = loss)
        mh_p = mh_p * (1-ψ) + mh_p * ψ
        mh_m = mh_m * (1-ψ) + mh_m * ψ
        qh_p = qh_p * (1-ψ) + qh_p * ψ
        qh_m = qh_m * (1-ψ) + qh_m * ψ
        Vh_p = Vh_p * (1-ψ) + Vh_p * ψ
        Vh_m = Vh_m * (1-ψ) + Vh_m * ψ
        m_p = dmh_p_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m)
        m_m = dmh_m_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m)
        q_p = -2*dVh_p_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m)
        q_m = -2*dVh_m_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m)
        V_p = 2*dqh_p_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m)
        V_m = 2*dqh_m_Gs(mh_p, mh_m, qh_p, qh_m, Vh_p, Vh_m, λ, Σp, Σm, μ_p, μ_m)
        m_p = m_p * (1-ψ) + m_p_init * ψ
        m_m = m_m * (1-ψ) + m_m_init * ψ
        q_p = q_p * (1-ψ) + q_p_init * ψ
        q_m = q_m * (1-ψ) + q_m_init * ψ
        V_p = V_p * (1-ψ) + V_p_init * ψ
        V_m = V_m * (1-ψ) + V_m_init * ψ
        Δ = (np.abs(m_p - m_p_init) + np.abs(m_m - m_m_init) + np.abs(q_p - q_p_init) + np.abs(q_m - q_m_init) + np.abs(V_p - V_p_init) + np.abs(V_m - V_m_init))/6
        if Δ < ϵ and it > 5:
            ok = "true"
            break
        m_p_init = m_p
        m_m_init = m_m
        q_p_init = q_p
        q_m_init = q_m
        V_p_init = V_p
        V_m_init = V_m
    return m_p, m_m, q_p, q_m, V_p, V_m, ok

def get_gaussian_mixture_learning_curve(μ_p, Σp, μ_m, Σm, mean_identifier, cov_identifier, αs=[0.1], λ=0.01, loss = "square_loss", ψ=0., ρ_p =0.5, ρ_m = 0.5,
                                        path_to_res_folder = "./", resfile = "res.csv", max_iters = 1000, ϵ = 1e-7):

    res = {'alpha': [], 'train_loss': [], 'p': [], 'mean_identifier': [], 'cov_identifier':[], 'loss': [],
           'ρ_p': [], 'ρ_m': [], 'lamb': [], 'm_p':[], 'm_m':[], 'q_p': [], 'q_m': [], 'V_p': [], 'V_m': [], 'ok':[]}
    p = len(μ_p)
    m_p_init = 0.1; m_m_init = 0.1; q_p_init = 0.2; q_m_init = 0.2; V_p_init = 100.; V_m_init = 100.
    tl = 0.
    for α in αs:
        print("-----------------------> processing n/p = ", α)
        m_p, m_m, q_p, q_m, V_p, V_m, ok = iterate_saddle_point_equations(m_p_init, m_m_init, q_p_init, q_m_init, V_p_init, V_m_init, Σp, Σm, μ_p, μ_m, ρ_p = ρ_p, ρ_m = ρ_m, α = α, λ = λ, max_iters = max_iters, ϵ = ϵ,
                                                  ψ = ψ, loss = loss)
        tl = train_loss(m_p, m_m, q_p, q_m, V_p, V_m, ρ_p, ρ_m, loss = loss)
        res["p"].append(p)
        res["alpha"].append(α)
        res["mean_identifier"].append(mean_identifier)
        res["cov_identifier"].append(cov_identifier)
        res["loss"].append(loss)
        res["ρ_p"].append(ρ_p)
        res["ρ_m"].append(ρ_m)
        res["lamb"].append(λ)
        res["m_p"].append(m_p)
        res["m_m"].append(m_m)
        res["q_p"].append(q_p)
        res["q_m"].append(q_m)
        res["V_p"].append(V_p)
        res["V_m"].append(V_m)
        res["train_loss"].append(tl)
        res["ok"].append(ok)
        m_p_init = m_p
        m_m_init = m_m
        q_p_init = q_p
        q_m_init = q_m
        V_p_init = V_p
        V_m_init = V_m

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
