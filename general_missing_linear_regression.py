import argparse
import concurrent.futures
import json
import os
import warnings
from collections import Counter
from functools import partial

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)

from rpy2 import robjects
from rpy2.robjects import globalenv, pandas2ri
from rpy2.robjects.packages import importr

pandas2ri.activate()
robjects.r('library("mice")')
robjects.r('set.seed(42)')
r_mice = robjects.r['mice']
r_complete = robjects.r['complete']
r_make_predictorMatrix = robjects.r['make.predictorMatrix']


class DataSimulatorCase1(object):
    def __init__(self, N, beta_0, beta_1, beta_2, sigma, scale, use_estimated_ps=False, prediction_noise=0, prediction_bias=0, missing_mechanism=None, mispecified_ps_type=0):
        self.N = N
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.sigma = sigma
        self.scale = scale
        self.use_estimated_ps = use_estimated_ps
        self.prediction_noise = prediction_noise
        self.prediction_bias = prediction_bias
        self.missing_mechanism = missing_mechanism
        self.mispecified_ps_type = mispecified_ps_type

    def generate(self) -> np.ndarray:
        mean = [0, 0]
        cov = [
            [self.scale**2, 0.4 * self.scale * self.scale],
            [0.4 * self.scale * self.scale, self.scale**2]
        ]

        # Generate samples from the bivariate normal distribution
        Z1, Z2 = np.random.multivariate_normal(mean, cov, self.N).T
        Z = np.stack([Z1, Z2], axis=1)  # shape (n, 2)

        X1 = 0.1 * np.exp(Z1) + np.random.randn(self.N) * 1.5 * self.scale
        X2 = np.sin(Z2) + np.random.exponential(0.1 * self.scale, self.N)
        X = np.stack([np.ones(self.N), X1, X2], axis=1)

        epsilon = self.sigma * np.random.randn(self.N)
        y = self.beta_0 + self.beta_1 * X1 + self.beta_2 * X2 + epsilon

        predicted_X1 = X1 + np.random.exponential(self.prediction_bias, size=self.N) + self.prediction_noise * np.random.randn(self.N)
        predicted_X2 = X2 + np.random.exponential(self.prediction_bias, size=self.N) + self.prediction_noise * np.random.randn(self.N)
        predicted_X = np.stack([np.ones(self.N), predicted_X1, predicted_X2], axis=1)  # shape (n, 3)
        predicted_y = y + np.random.exponential(self.prediction_bias, size=self.N) + self.prediction_noise * np.random.randn(self.N)

        #   Y X1 X2 
        # 2: X 0 0
        # 3: 0 0 X
        # 4: X X 0
        coarseningIndicator = []
        if self.missing_mechanism == "MAR":
            raw_propensity_score = np.stack([
                np.zeros(self.N),
                expit(-1.0 + 0.1 * X2 + 0.1 * Z1 + 0.1 * X1 * X2), 
                expit(-1.8 - 0.2 * y + 0.1 * X1 + 0.1 * Z1 + 0.3 * y * X1), 
                expit(-1.0 + 0.1 * X2 + 0.2 * Z1)
            ]).T
            raw_propensity_score[:, 0] = 1 - raw_propensity_score[:, 1:].sum(axis=1)

        elif self.missing_mechanism == "MCAR":
            raw_propensity_score = np.zeros((self.N, 4))
            raw_propensity_score[:, 0] = 0.05
            raw_propensity_score[:, 1] = 0.45
            raw_propensity_score[:, 2] = 0.35
            raw_propensity_score[:, 3] = 0.15
        else:
            raise NotImplementedError


        assert raw_propensity_score[:, 0].min() > 0
        propensity_score = raw_propensity_score

        for i in range(self.N):
            coarseningIndicator.append(np.random.choice([1, 2, 3, 4], size=1, p=propensity_score[i]))    
        coarseningIndicator = np.array(coarseningIndicator).squeeze()

        # Count occurrences
        counts = Counter(coarseningIndicator)
        
        y[coarseningIndicator == 2, ] = np.nan
        y[coarseningIndicator == 4, ] = np.nan
        X[coarseningIndicator == 3, 2] = np.nan
        X[coarseningIndicator == 4, 1] = np.nan
        
        if self.use_estimated_ps:
            if self.mispecified_ps_type == 0:
                estimated_propensity_score = self.fit_propensity_score_model(y, X, Z, coarseningIndicator)
            elif self.mispecified_ps_type == 1:
                estimated_propensity_score = self.fit_mispecified_propensity_score_model(Z, coarseningIndicator)
            propensity_score = estimated_propensity_score
        
        return X, y, predicted_X, predicted_y, coarseningIndicator, propensity_score, counts, Z

    def fit_propensity_score_model(self, y, X, Z, coarseningIndicator):
        # fit logistic regression to estimate propensity score for coarseningIndicator == 2, 3, 4
        # then coarseningIndicator == 1 equals to 1 - sum(coarseningIndicator == 2, 3, 4)

        # concatenate y and Z
        inputs = np.concatenate([y.reshape(-1, 1), X[:, 1:], Z, (X[:, 1] * X[:, 2]).reshape(-1, 1), (y * X[:, 1]).reshape(-1, 1)], axis=1)

        # coarseningIndicator == 2
        inputs_2 = inputs[coarseningIndicator == 2]
        covariates_2 = np.where(~np.isnan(inputs_2).any(axis=0))[0]
        inputs_2 = inputs_2[:, covariates_2]
        # add constant
        inputs_2 = sm.add_constant(inputs_2)

        # coarseningIndicator == 3
        inputs_3 = inputs[coarseningIndicator == 3]
        covariates_3 = np.where(~np.isnan(inputs_3).any(axis=0))[0]
        inputs_3 = inputs_3[:, covariates_3]
        # add constant
        inputs_3 = sm.add_constant(inputs_3)

        # coarseningIndicator == 4
        inputs_4 = inputs[coarseningIndicator == 4]
        covariates_4 = np.where(~np.isnan(inputs_4).any(axis=0))[0]
        inputs_4 = inputs_4[:, covariates_4]
        # add constant
        inputs_4 = sm.add_constant(inputs_4)

        # coarseningIndicator == 1
        inputs_1_2 = inputs[coarseningIndicator == 1][:, covariates_2]
        inputs_1_2 = sm.add_constant(inputs_1_2)
        inputs_1_3 = inputs[coarseningIndicator == 1][:, covariates_3]
        inputs_1_3 = sm.add_constant(inputs_1_3)
        inputs_1_4 = inputs[coarseningIndicator == 1][:, covariates_4]
        inputs_1_4 = sm.add_constant(inputs_1_4)

        num_of_params = inputs_2.shape[1] + inputs_3.shape[1] + inputs_4.shape[1]
        cumsum_num_of_params = np.cumsum([0, inputs_2.shape[1], inputs_3.shape[1], inputs_4.shape[1]])
        
        def likelihood(params):
            beta1 = params[:cumsum_num_of_params[1]]
            beta2 = params[cumsum_num_of_params[1]:cumsum_num_of_params[2]]
            beta3 = params[cumsum_num_of_params[2]:cumsum_num_of_params[3]]
            nll = 0
            
            nll += -np.sum(np.log(expit(inputs_2 @ beta1.T)))
            nll += -np.sum(np.log(expit(inputs_3 @ beta2.T)))
            nll += -np.sum(np.log(expit(inputs_4 @ beta3.T)))
            
            complete_ps = np.log(1 - expit(inputs_1_2 @ beta1.T) - expit(inputs_1_3 @ beta2.T) - expit(inputs_1_4 @ beta3.T))
            complete_ps = complete_ps[~np.isnan(complete_ps)]
            nll += -np.sum(complete_ps)
            
            return nll
        
        def constraint(index, params):
            beta1 = params[:cumsum_num_of_params[1]]
            beta2 = params[cumsum_num_of_params[1]:cumsum_num_of_params[2]]
            beta3 = params[cumsum_num_of_params[2]:cumsum_num_of_params[3]]

            return 0.99 - expit(inputs_1_2[index] @ beta1.T) - expit(inputs_1_3[index] @ beta2.T) - expit(inputs_1_4[index] @ beta3.T)


        initial_guess = -np.ones(num_of_params)

        constraints = [{'type': 'ineq', 'fun': partial(constraint, i)} for i in range(inputs[coarseningIndicator == 1].shape[0])]
        options = {'maxiter': 3000, 'tol': 1e-6, 'rhobeg': 0.1}
        result = minimize(likelihood, initial_guess, method='COBYLA', constraints=constraints, options=options)

        estimated_propensity_score = np.zeros((self.N, 4))
        estimated_propensity_score[coarseningIndicator == 1, 1] = expit(inputs_1_2 @ result.x[:cumsum_num_of_params[1]].T)
        estimated_propensity_score[coarseningIndicator == 1, 2] = expit(inputs_1_3 @ result.x[cumsum_num_of_params[1]:cumsum_num_of_params[2]].T)
        estimated_propensity_score[coarseningIndicator == 1, 3] = expit(inputs_1_4 @ result.x[cumsum_num_of_params[2]:cumsum_num_of_params[3]].T)
        estimated_propensity_score[coarseningIndicator == 1, 0] = 1 - estimated_propensity_score[coarseningIndicator == 1, 1:].sum(axis=1)
        
        estimated_propensity_score[coarseningIndicator == 2, 1] = expit(inputs_2 @ result.x[:cumsum_num_of_params[1]].T)
        estimated_propensity_score[coarseningIndicator == 3, 2] = expit(inputs_3 @ result.x[cumsum_num_of_params[1]:cumsum_num_of_params[2]].T)
        estimated_propensity_score[coarseningIndicator == 4, 3] = expit(inputs_4 @ result.x[cumsum_num_of_params[2]:cumsum_num_of_params[3]].T)

        assert (estimated_propensity_score[coarseningIndicator == 1, 0] > 0).all()

        return estimated_propensity_score

    def fit_mispecified_propensity_score_model(self, Z, coarseningIndicator):
        # fit logistic regression to estimate propensity score for coarseningIndicator == 2, 3, 4
        # then coarseningIndicator == 1 equals to 1 - sum(coarseningIndicator == 2, 3, 4)

        # concatenate y and Z
        inputs = Z

        covariates = np.array([0])

        # coarseningIndicator == 2
        inputs_2 = inputs[coarseningIndicator == 2]
        inputs_2 = inputs_2[:, covariates]
        # add constant
        inputs_2 = sm.add_constant(inputs_2)

        # coarseningIndicator == 3
        inputs_3 = inputs[coarseningIndicator == 3]
        inputs_3 = inputs_3[:, covariates]
        # add constant
        inputs_3 = sm.add_constant(inputs_3)

        # coarseningIndicator == 4
        inputs_4 = inputs[coarseningIndicator == 4]
        inputs_4 = inputs_4[:, covariates]
        # add constant
        inputs_4 = sm.add_constant(inputs_4)

        # coarseningIndicator == 1
        inputs_1_2 = inputs[coarseningIndicator == 1][:, covariates]
        inputs_1_2 = sm.add_constant(inputs_1_2)
        inputs_1_3 = inputs[coarseningIndicator == 1][:, covariates]
        inputs_1_3 = sm.add_constant(inputs_1_3)
        inputs_1_4 = inputs[coarseningIndicator == 1][:, covariates]
        inputs_1_4 = sm.add_constant(inputs_1_4)
        
        num_of_params = inputs_2.shape[1] + inputs_3.shape[1] + inputs_4.shape[1]
        cumsum_num_of_params = np.cumsum([0, inputs_2.shape[1], inputs_3.shape[1], inputs_4.shape[1]])
        
        def likelihood(params):
            beta1 = params[:cumsum_num_of_params[1]]
            beta2 = params[cumsum_num_of_params[1]:cumsum_num_of_params[2]]
            beta3 = params[cumsum_num_of_params[2]:cumsum_num_of_params[3]]
            nll = 0
            
            nll += -np.sum(np.log(expit(inputs_2 @ beta1.T)))
            nll += -np.sum(np.log(expit(inputs_3 @ beta2.T)))
            nll += -np.sum(np.log(expit(inputs_4 @ beta3.T)))
            
            complete_ps = np.log(1 - expit(inputs_1_2 @ beta1.T) - expit(inputs_1_3 @ beta2.T) - expit(inputs_1_4 @ beta3.T))
            complete_ps = complete_ps[~np.isnan(complete_ps)]
            nll += -np.sum(complete_ps)
            
            return nll
        
        def constraint(index, params):
            beta1 = params[:cumsum_num_of_params[1]]
            beta2 = params[cumsum_num_of_params[1]:cumsum_num_of_params[2]]
            beta3 = params[cumsum_num_of_params[2]:cumsum_num_of_params[3]]

            return 0.99 - expit(inputs_1_2[index] @ beta1.T) - expit(inputs_1_3[index] @ beta2.T) - expit(inputs_1_4[index] @ beta3.T)


        initial_guess = -np.ones(num_of_params)

        constraints = [{'type': 'ineq', 'fun': partial(constraint, i)} for i in range(inputs[coarseningIndicator == 1].shape[0])]
        options = {'maxiter': 1000, 'tol': 1e-6, 'rhobeg': 0.1}
        result = minimize(likelihood, initial_guess, method='COBYLA', constraints=constraints, options=options)

        estimated_propensity_score = np.zeros((self.N, 4))
        estimated_propensity_score[coarseningIndicator == 1, 1] = expit(inputs_1_2 @ result.x[:cumsum_num_of_params[1]].T)
        estimated_propensity_score[coarseningIndicator == 1, 2] = expit(inputs_1_3 @ result.x[cumsum_num_of_params[1]:cumsum_num_of_params[2]].T)
        estimated_propensity_score[coarseningIndicator == 1, 3] = expit(inputs_1_4 @ result.x[cumsum_num_of_params[2]:cumsum_num_of_params[3]].T)
        estimated_propensity_score[coarseningIndicator == 1, 0] = 1 - estimated_propensity_score[coarseningIndicator == 1, 1:].sum(axis=1)
        
        estimated_propensity_score[coarseningIndicator == 2, 1] = expit(inputs_2 @ result.x[:cumsum_num_of_params[1]].T)
        estimated_propensity_score[coarseningIndicator == 3, 2] = expit(inputs_3 @ result.x[cumsum_num_of_params[1]:cumsum_num_of_params[2]].T)
        estimated_propensity_score[coarseningIndicator == 4, 3] = expit(inputs_4 @ result.x[cumsum_num_of_params[2]:cumsum_num_of_params[3]].T)

        assert (estimated_propensity_score[coarseningIndicator == 1, 0] > 0).all()

        return estimated_propensity_score

def jackknife_covariance(estimates):
    estimates = np.array(estimates)
    
    n = estimates.shape[0]
    estimates_mean = np.mean(estimates, axis=0)
    diffs = estimates - estimates_mean
    cov_matrix = (n - 1) / n * np.dot(diffs.T, diffs)

    return cov_matrix


def numpy2pandas(X, y, Z):
    X_df = pd.DataFrame(X, columns=['intercept', 'x1', 'x2'])
    y_df = pd.DataFrame(y, columns=['y'])
    Z_df = pd.DataFrame(Z, columns=['z1', 'z2'])
    dataset = pd.concat([X_df, y_df, Z_df], axis=1)
    return dataset
    

def multiple_imputation_linear_regression(X, y, Z, m=5, method="pmm", seed=42):
    """
    Perform multiple imputation linear regression using R's mice package via rpy2.
    Automatically constructs the formula y ~ all other columns in df,
    then runs: mice() → with(lm) → pool() → summary() all inside R.

    Parameters
    ----------
    X : array‑like, shape (n_samples, n_features)
        Predictor matrix.
    y : array‑like, shape (n_samples,)
        Outcome vector.
    m : int, default=5
        Number of imputations.
    method : str, default="pmm"
        mice imputation method.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    estimate_arr : numpy.ndarray
        Pooled point estimates for each term.
    variance_arr : numpy.ndarray
        Pooled total variances for each term.
    p_value_arr : numpy.ndarray
        Two‑sided p‑values under H0: β = 0.
    """

    # 1) Convert X, y → pandas DataFrame with column 'y'
    df = numpy2pandas(X, y, Z)

    # 2) Push into R, enable pandas<->R bridge
    pandas2ri.activate()
    globalenv['df'] = pandas2ri.py2rpy(df)

    # 3) Run the entire pipeline in R
    #    - load mice
    #    - set seed
    #    - build formula = y ~ all other cols
    #    - impute, fit, pool, summarize
    r_script = f"""
    suppressMessages(library(mice))
    set.seed({seed})

    imp     <- mice(df, m = {m}, method = "{method}", printFlag = FALSE)
    fit_imp <- with(imp, lm(y ~ x1 + x2))
    pooled  <- pool(fit_imp)

    # ask for 95% CIs and rename them to lb/ub
    sum_tab <- summary(pooled, conf.int = TRUE, conf.level = 0.95)
    colnames(sum_tab)[colnames(sum_tab) == "2.5 %"]  <- "lb"
    colnames(sum_tab)[colnames(sum_tab) == "97.5 %"] <- "ub"
    """
    robjects.r(r_script)

    # 4) Grab the summary table back into Python
    sum_tab = robjects.globalenv['sum_tab']

    # 4) Bring the summary table back into Python
    res_df  = pandas2ri.rpy2py(sum_tab)

    # 5) Extract NumPy arrays
    estimate_arr = res_df['estimate'].to_numpy()
    variance_arr = (res_df['std.error'] ** 2).to_numpy()
    lb_arr = res_df['lb'].to_numpy()
    ub_arr = res_df['ub'].to_numpy()

    return estimate_arr, variance_arr, lb_arr, ub_arr

def pinv(a):
    return np.linalg.pinv(a)

def get_exog_and_resid(res_fit):
    X = np.asarray(res_fit.model.exog, dtype=float)
    r = np.asarray(res_fit.resid, dtype=float).reshape(-1)
    return X, r

def bread_from_fit(res_fit, weights=None):
    X, _ = get_exog_and_resid(res_fit)
    N = X.shape[0]

    if weights is not None:
        return X.T @ (weights[:, None] * X)
    else:
        return X.T @ X

def meat_from_fit(res_fit, weights=None):
    X, r = get_exog_and_resid(res_fit)
    N = X.shape[0]
    
    if weights is not None:
        r = r * weights

    return (X * r[:, None]).T @ (X * r[:, None])

def cross_meat_from_fits(res_a, res_b, weights=None):
    Xa, ra = get_exog_and_resid(res_a)
    Xb, rb = get_exog_and_resid(res_b)
    if Xa.shape[0] != Xb.shape[0]:
        raise ValueError("cross_meat_from_fits expects fits on the SAME rows (same N).")
    N = Xa.shape[0]

    if weights is not None:
        ra = ra * weights
        rb = rb * weights

    return (Xa * ra[:, None]).T @ (Xb * rb[:, None])

def compute_psppi_covariance(
    theta, gamma_1s, gamma_2s, 
    complete_data_weights=None, incomplete_data_weights=None
):
    if incomplete_data_weights is not None:
        assert isinstance(incomplete_data_weights, dict)
    # ---------------------------
    # 1) Dimensions and basic sandwich for theta
    # ---------------------------
    p = len(list(theta.params))

    B_theta = bread_from_fit(theta, weights=complete_data_weights)
    M_theta = meat_from_fit(theta, weights=complete_data_weights)
    B_theta_inv = pinv(B_theta)
    Sigma_theta = B_theta_inv @ M_theta @ B_theta_inv.T  # (p x p)

    # ---------------------------
    # 2) Inverses of breads for γ1_k and γ2_k, and sizes p_k
    # ---------------------------
    B_g1_inv = {}
    B_g2_inv = {}

    # Fix a deterministic order of patterns to assemble block matrices
    patt_order = list(gamma_1s.keys())

    for patt in patt_order:
        B_g1_inv[patt] = pinv(bread_from_fit(gamma_1s[patt], weights=complete_data_weights))
        if incomplete_data_weights is not None:
            B_g2_inv[patt] = pinv(bread_from_fit(gamma_2s[patt], weights=incomplete_data_weights[patt]))
        else:
            B_g2_inv[patt] = pinv(bread_from_fit(gamma_2s[patt]))
    # ---------------------------
    # 3) Build Σ_{θ,γ1·} (concatenate across patterns by columns)
    #     Σ_{θ,γ1k} = B_θ^{-1} M_cross(θ,γ1k) B_{γ1k}^{-T}
    # ---------------------------
    Sigma_theta_g1 = {}
    for patt in patt_order:
        g1 = gamma_1s[patt]
        # Cross-meat over the SAME rows (complete_data), weighted by 1/pi_infty if provided
        M_cross = cross_meat_from_fits(theta, g1, weights=complete_data_weights)  # shape p x p_k
        Sigma_theta_g1[patt] = B_theta_inv @ M_cross @ B_g1_inv[patt].T  # p x p_k

    # ---------------------------
    # 4) Build Σ_{γ1γ1} as a full (q x q) block matrix over all (k,k')
    #     Σ_{γ1k,γ1k'} = B_{γ1k}^{-1} M_cross(γ1k,γ1k') B_{γ1k'}^{-T}
    # ---------------------------
    Sigma_g1g1 = {}
    for pk in patt_order:
        for pkp in patt_order:
            g1k, g1kp = gamma_1s[pk], gamma_1s[pkp]
            M_g1k_g1kp = cross_meat_from_fits(g1k, g1kp, weights=complete_data_weights) 
            Sigma_g1g1[(pk, pkp)] = B_g1_inv[pk] @ M_g1k_g1kp @ B_g1_inv[pkp].T  # p_k x p_k'

    # ---------------------------
    # 5) Build Σ_{γ2} as block-diagonal (q x q), aligned with patt_order
    #     Σ_{γ2k} = B_{γ2k}^{-1} M(γ2k) B_{γ2k}^{-T}
    # ---------------------------
    Sigma_g2 = {}
    for patt in patt_order:
        g2 = gamma_2s[patt]
        if incomplete_data_weights is not None:
            M2 = meat_from_fit(g2, weights=incomplete_data_weights[patt])            # p_k x p_k
        else:
            M2 = meat_from_fit(g2)
        Sigma_g2[patt] = B_g2_inv[patt] @ M2 @ B_g2_inv[patt].T

    # ---------------------------
    # 6) Assemble Σ_{γγ} and compute W and Σ_{PS-PPI}
    # ---------------------------
    sum_Sigma_theta_g1 = np.zeros(Sigma_theta.shape)
    for patt, Sigma_theta_g1k in Sigma_theta_g1.items():
        sum_Sigma_theta_g1 += Sigma_theta_g1k

    sum_Sigma_g1g1 = np.zeros(Sigma_theta.shape)
    for patt, Sigma_g1k_g1kp in Sigma_g1g1.items():
        sum_Sigma_g1g1 += Sigma_g1k_g1kp

    sum_Sigma_g2 = np.zeros(Sigma_theta.shape)
    for patt, Sigma_g2k in Sigma_g2.items():
        sum_Sigma_g2 += Sigma_g2k

    W = sum_Sigma_theta_g1 @ np.linalg.pinv(sum_Sigma_g1g1 + sum_Sigma_g2)
    Sigma_psppi = Sigma_theta - sum_Sigma_theta_g1 @ np.linalg.pinv(sum_Sigma_g1g1 + sum_Sigma_g2) @ sum_Sigma_theta_g1.T

    theta_psppi = theta.params
    for patt in patt_order:
        theta_psppi = theta_psppi - W @ (gamma_1s[patt].params - gamma_2s[patt].params)

    return {
        "W": W,
        "Sigma_psppi": Sigma_psppi,
        "Sigma_theta": Sigma_theta,
        "theta_psppi": theta_psppi,
    }


def run_simulation(seed, args, data_simulator, method):
    # Simulate data
    np.random.seed(seed + 2025)

    N = args.N
    beta_0 = args.beta_0
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    quantile = norm.ppf(0.975)

    X, y, predicted_X, predicted_y, coarseningIndicator, propensity_score, counts, Z = data_simulator.generate()

    output_deltas = None
    output_theta = None
    all_params_list = {}
    all_var_list = {}
    if method == "CCA":
        labeled_X = X[coarseningIndicator == 1]
        labeled_y = y[coarseningIndicator == 1]

        model = sm.OLS(labeled_y, labeled_X)
        results = model.fit(cov_type=args.cov_type)
        params = results.params
        total_var = np.diag(results.cov_params())
        
        half_width = quantile * np.sqrt(total_var)
        ub = params + half_width
        lb = params - half_width
    elif method == "WCCA":
        labeled_X = X[coarseningIndicator == 1]
        labeled_y = y[coarseningIndicator == 1]
        labeled_propensity_score = propensity_score[coarseningIndicator == 1][:, 0]

        model = sm.WLS(labeled_y, labeled_X, weights=1 / labeled_propensity_score)
        results = model.fit(cov_type=args.cov_type)
        params = results.params
        total_var = np.diag(results.cov_params())
        
        half_width = quantile * np.sqrt(total_var)
        ub = params + half_width
        lb = params - half_width
    elif method == "MI":
        params, total_var, lb, ub = multiple_imputation_linear_regression(X, y, Z)
        half_width = (ub - lb) / 2
    elif method == "PPI++":
        from ppi_py import ppi_ols_pointestimate, ppi_ols_ci
        y_labeled = y[coarseningIndicator == 1]
        y_hat_labeled = predicted_y[coarseningIndicator == 1]
        X_labeled = X[coarseningIndicator == 1]

        X_unlabeled = X[coarseningIndicator == 2]
        y_hat_unlabeled = predicted_y[coarseningIndicator == 2]

        params = ppi_ols_pointestimate(X=X_labeled, Y=y_labeled, Yhat=y_hat_labeled, X_unlabeled=X_unlabeled, Yhat_unlabeled=y_hat_unlabeled)
        lb, ub = ppi_ols_ci(X=X_labeled, Y=y_labeled, Yhat=y_hat_labeled, X_unlabeled=X_unlabeled, Yhat_unlabeled=y_hat_unlabeled, alpha=0.05)
        half_width = (ub - lb) / 2
    elif method in ["PPI_reimp", "PPI++_reimp"]:  # PS-PPI (outcome only)
        # extract subset of coarseningIndicator == 1
        X_complete_cases = X[coarseningIndicator == 1].copy()
        y_complete_cases = y[coarseningIndicator == 1].copy()
        predicted_X_complete_cases = predicted_X[coarseningIndicator == 1].copy()
        predicted_y_complete_cases = predicted_y[coarseningIndicator == 1].copy()
        propensity_score_complete_cases = propensity_score[coarseningIndicator == 1].copy()
        
        # generate pseudo coarsening patterns for complete data
        pseudo_coarseningIndicator = []
        for scores in propensity_score_complete_cases[:, 1:]:
            pseudo_coarseningIndicator.append(2)
        pseudo_coarseningIndicator = np.array(pseudo_coarseningIndicator).squeeze()

        # imput missing value with machine learning predictions
        imputed_labeled_X = X[coarseningIndicator == 1].copy()
        imputed_labeled_y = y[coarseningIndicator == 1].copy()
        for coarsening_idx in range(2, 3):
            if coarsening_idx == 2:
                imputed_labeled_y[pseudo_coarseningIndicator == coarsening_idx] = predicted_y_complete_cases[pseudo_coarseningIndicator == coarsening_idx]

        theta = []
        gamma1 = []
        for i in range(y_complete_cases.shape[0]):
            X_input_drop_i =  np.delete(X_complete_cases, i, axis=0)
            y_input_drop_i = np.delete(y_complete_cases, i, axis=0)
            pseudo_coarseningIndicator_drop_i = np.delete(pseudo_coarseningIndicator, i, axis=0)
            propensity_score_input_drop_i_complete = np.delete(propensity_score_complete_cases, i, axis=0)[:, 0]
            propensity_score_input_drop_i_pseudo = np.delete(propensity_score_complete_cases, i, axis=0)[range(X_input_drop_i.shape[0]), pseudo_coarseningIndicator_drop_i - 1]
            propensity_score_input_drop_i = propensity_score_input_drop_i_complete * propensity_score_input_drop_i_pseudo

            imputed_X_input_drop_i = np.delete(imputed_labeled_X, i, axis=0)
            imputed_y_input_drop_i = np.delete(imputed_labeled_y, i, axis=0)
            
            # use OLS
            model = sm.OLS(y_input_drop_i, X_input_drop_i)
            results = model.fit()
            theta.append(results.params)
            
            model = sm.OLS(imputed_y_input_drop_i, imputed_X_input_drop_i)
            results = model.fit()
            gamma1.append(results.params)
            
        # impute real missing data
        unlabeled_X = X[coarseningIndicator == 2]
        unlabeled_y = y[coarseningIndicator == 2]
        unlabeled_predicted_X = predicted_X[coarseningIndicator == 2]
        unlabeled_predicted_y = predicted_y[coarseningIndicator == 2]
        unlabeled_coarseningIndicator = coarseningIndicator[coarseningIndicator == 2]
        unlabeled_propensity_score = propensity_score[coarseningIndicator == 2]
      
        imputed_unlabeled_X = unlabeled_X.copy()
        imputed_unlabeled_y = unlabeled_y.copy()
        for coarsening_idx in range(2, 3):
            if coarsening_idx == 2:
                imputed_unlabeled_y[unlabeled_coarseningIndicator == coarsening_idx] = unlabeled_predicted_y[unlabeled_coarseningIndicator == coarsening_idx]

        model = sm.OLS(imputed_unlabeled_y, imputed_unlabeled_X)
        results = model.fit(cov_type=args.cov_type)
        gamma2 = results.params
        gamma2_var = results.cov_params()
        
        theta = np.array(theta)  # shape (n, 3)
        gamma1 = np.array(gamma1)  # shape (n, 3)
        theta_gamma1 = np.concatenate([theta, gamma1], axis=1)  # shape (n, 6)
        
        theta_gamma1_var = jackknife_covariance(theta_gamma1)  # shape (6, 6)
        # reshape into 2 x 2 x 3 x 3
        theta_gamma1_var_reshaped = theta_gamma1_var.reshape(2, 3, 2, 3).transpose(0, 2, 1, 3)
        
        if method == "PPI++_reimp":
            W = theta_gamma1_var_reshaped[0, 1] + theta_gamma1_var_reshaped[0, 1] @ np.linalg.inv(theta_gamma1_var_reshaped[1, 1] + gamma2_var)
            total_var = theta_gamma1_var_reshaped[0, 0] - theta_gamma1_var_reshaped[0, 1] @ np.linalg.inv(theta_gamma1_var_reshaped[1, 1] + gamma2_var) @ theta_gamma1_var_reshaped[0, 1].T
            total_var = np.diag(total_var)
            params = np.mean(theta, axis=0) - W @ (np.mean(gamma1, axis=0) - gamma2)
        elif method == "PPI_reimp":
            W = np.eye(3)
            total_var = np.zeros((3, 3))
            total_var = theta_gamma1_var_reshaped[0, 0] + theta_gamma1_var_reshaped[1, 1] - theta_gamma1_var_reshaped[0, 1] - theta_gamma1_var_reshaped[1, 0] + gamma2_var
            total_var = np.diag(total_var)
            params = np.mean(theta, axis=0) - W @ (np.mean(gamma1, axis=0) - gamma2)
        
        half_width = quantile * np.sqrt(total_var)
        if np.isnan(half_width).sum() > 0:
            print(total_var)
            print(W)
        ub = params + half_width
        lb = params - half_width
        
        output_deltas = np.mean(gamma1, axis=0) - gamma2
        output_theta = np.mean(theta, axis=0)

    elif method == "PS-PPI":
        gamma_2j = []
        gamma_2j_var = []
        for coarsening_idx in range(2, 5):
            if coarsening_idx == 2:
                X_input = X[coarseningIndicator == coarsening_idx]
                y_input = predicted_y[coarseningIndicator == coarsening_idx]
                propensity_score_input = propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]
            elif coarsening_idx == 3:
                X_input = X[coarseningIndicator == coarsening_idx]
                X_input[:, 2] = predicted_X[coarseningIndicator == coarsening_idx][:, 2]
                y_input = y[coarseningIndicator == coarsening_idx]
                propensity_score_input = propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]
            elif coarsening_idx == 4:
                X_input = X[coarseningIndicator == coarsening_idx]
                X_input[:, 1] = predicted_X[coarseningIndicator == coarsening_idx][:, 1]
                y_input = predicted_y[coarseningIndicator == coarsening_idx]
                propensity_score_input = propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]

            # use sm.WLS
            model = sm.WLS(y_input, X_input, weights=1 / propensity_score_input)
            results = model.fit(cov_type=args.cov_type)
            gamma_2j.append(results.params)
            gamma_2j_var.append(results.cov_params())
        
        # extract subset of coarseningIndicator == 1
        X_complete_cases = X[coarseningIndicator == 1]
        y_complete_cases = y[coarseningIndicator == 1]  
        predicted_X_complete_cases = predicted_X[coarseningIndicator == 1]
        predicted_y_complete_cases = predicted_y[coarseningIndicator == 1]
        propensity_score_complete_cases = propensity_score[coarseningIndicator == 1][:, 0]
        
        # jackknife
        theta = []
        gamma_1j = []
        for i in range(y_complete_cases.shape[0]):
            gamma_1j_drop_i = []
            propensity_score_drop_i = np.delete(propensity_score_complete_cases, i, axis=0)  
            model = sm.WLS(np.delete(y_complete_cases, i, axis=0), np.delete(X_complete_cases, i, axis=0), weights=1 / propensity_score_drop_i)
            results = model.fit()
            theta.append(results.params)
            for coarsening_idx in range(2, 5):
                if coarsening_idx == 2:
                    X_input_drop_i =  np.delete(X_complete_cases, i, axis=0)
                    y_input_drop_i = np.delete(predicted_y_complete_cases, i, axis=0)
                    model = sm.WLS(y_input_drop_i, X_input_drop_i, weights=1 / propensity_score_drop_i)
                    results = model.fit()
                    gamma_1j_drop_i.append(results.params)
                elif coarsening_idx == 3:
                    X_input_drop_i =  np.delete(X_complete_cases, i, axis=0)
                    X_input_drop_i[:, 2] = np.delete(predicted_X_complete_cases, i, axis=0)[:, 2]
                    y_input_drop_i = np.delete(y_complete_cases, i, axis=0)
                    model = sm.WLS(y_input_drop_i, X_input_drop_i, weights=1 / propensity_score_drop_i)
                    results = model.fit()
                    gamma_1j_drop_i.append(results.params)
                elif coarsening_idx == 4:
                    X_input_drop_i =  np.delete(X_complete_cases, i, axis=0)
                    X_input_drop_i[:, 1] = np.delete(predicted_X_complete_cases, i, axis=0)[:, 1]
                    y_input_drop_i = np.delete(predicted_y_complete_cases, i, axis=0)
                    model = sm.WLS(y_input_drop_i, X_input_drop_i, weights=1 / propensity_score_drop_i)
                    results = model.fit()
                    gamma_1j_drop_i.append(results.params)
            gamma_1j.append(gamma_1j_drop_i)
        
        theta = np.array(theta)
        gamma_1j = np.array(gamma_1j)
        theta_gamma_1j = np.concatenate([theta, np.array(gamma_1j).reshape(-1, 9)], axis=1)  # TODO: check the reshape operation

        assert theta.shape == (y_complete_cases.shape[0], 3)
        assert gamma_1j.shape == (y_complete_cases.shape[0], 3, 3)
        assert theta_gamma_1j.shape == (y_complete_cases.shape[0], 3 * 4)
        
        theta_gamma_1j_var = jackknife_covariance(theta_gamma_1j)
        # reshape into 4 x 4 x 3 x 3
        theta_gamma_1j_var_reshaped = theta_gamma_1j_var.reshape(4, 3, 4, 3).transpose(0, 2, 1, 3)

        total_var = np.zeros((3, 3))
        W1 = np.zeros((3, 3))
        W2 = np.zeros((3, 3))
        for i in range(3):
            W1 += theta_gamma_1j_var_reshaped[0, 1 + i]  # [:3, 3 + i * 3:3 + (i + 1) * 3]
            W2 += gamma_2j_var[i]
            for j in range(3):
                W2 += theta_gamma_1j_var_reshaped[1 + i, 1 + j]  # [3 + i * 3:3 + (i + 1) * 3, 3 + j * 3:3 + (j + 1) * 3]

        W2_inv = np.linalg.pinv(W2)
        W = np.dot(W1, W2_inv)
        total_var += theta_gamma_1j_var_reshaped[0, 0]  # theta_var  # [:3, :3]
        total_var -= W1 @ W2_inv @ W1.T
        total_var = np.diag(total_var)

        gamma_1j = np.mean(gamma_1j, axis=0)
        gamma_2j = np.array(gamma_2j)

        deltas_group = gamma_1j - gamma_2j
        deltas = np.zeros(deltas_group.shape[1])
        for i in range(Ws.shape[0]):
            deltas += Ws[i] @ deltas_group[i]

        theta = np.mean(theta, axis=0)
        params = theta - W @ deltas

        half_width = quantile * np.sqrt(total_var)
        ub = params + half_width
        lb = params - half_width

        if np.isnan(half_width).sum() > 0:
            print(total_var)
        
    elif method == "PS-PPI_closedform":
        gamma_2j = {}
        for coarsening_idx in range(2, 5):
            if coarsening_idx == 2:
                X_input = X[coarseningIndicator == coarsening_idx]
                y_input = predicted_y[coarseningIndicator == coarsening_idx]
                propensity_score_input = propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]
            elif coarsening_idx == 3:
                X_input = X[coarseningIndicator == coarsening_idx]
                X_input[:, 2] = predicted_X[coarseningIndicator == coarsening_idx][:, 2]
                y_input = y[coarseningIndicator == coarsening_idx]
                propensity_score_input = propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]
            elif coarsening_idx == 4:
                X_input = X[coarseningIndicator == coarsening_idx]
                X_input[:, 1] = predicted_X[coarseningIndicator == coarsening_idx][:, 1]
                y_input = predicted_y[coarseningIndicator == coarsening_idx]
                propensity_score_input = propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]

            # use sm.WLS
            model = sm.WLS(y_input, X_input, weights=1 / propensity_score_input)
            results = model.fit(cov_type=args.cov_type)
            gamma_2j[coarsening_idx] = results

        # extract subset of coarseningIndicator == 1
        X_complete_cases = X[coarseningIndicator == 1]
        y_complete_cases = y[coarseningIndicator == 1]  
        predicted_X_complete_cases = predicted_X[coarseningIndicator == 1]
        predicted_y_complete_cases = predicted_y[coarseningIndicator == 1]
        propensity_score_complete_cases = propensity_score[coarseningIndicator == 1][:, 0]

        gamma_1j = {}
        model = sm.WLS(y_complete_cases, X_complete_cases, weights=1 / propensity_score_complete_cases)
        theta = model.fit()
        for coarsening_idx in range(2, 5):
            if coarsening_idx == 2:
                model = sm.WLS(predicted_y_complete_cases, X_complete_cases, weights=1 / propensity_score_complete_cases)
                results = model.fit()
            elif coarsening_idx == 3:
                X_input = X_complete_cases.copy()
                X_input[:, 2] = predicted_X_complete_cases[:, 2]
                model = sm.WLS(y_complete_cases, X_input, weights=1 / propensity_score_complete_cases)
                results = model.fit()
            elif coarsening_idx == 4:
                X_input = X_complete_cases.copy()
                X_input[:, 1] = predicted_X_complete_cases[:, 1]
                model = sm.WLS(predicted_y_complete_cases, X_input, weights=1 / propensity_score_complete_cases)
                results = model. fit()
            else:
                raise ValueError("Invalid coarsening index")
            gamma_1j[coarsening_idx] = results

        incomplete_data_weights = {}
        for coarsening_idx in range(2, 5):
            incomplete_data_weights[coarsening_idx] = 1 / propensity_score[coarseningIndicator == coarsening_idx][:, coarsening_idx - 1]

        results = compute_psppi_covariance(theta, gamma_1j, gamma_2j, complete_data_weights= 1 / propensity_score_complete_cases, incomplete_data_weights= incomplete_data_weights)
        total_var = np.diag(results["Sigma_psppi"])
        params = results["theta_psppi"]

        half_width = quantile * np.sqrt(total_var)
        ub = params + half_width
        lb = params - half_width

        if np.isnan(half_width).sum() > 0:
            print(total_var)

    # determine whether the truth is within confidence intervals
    beta0_coverage = 0
    beta1_coverage = 0
    beta2_coverage = 0

    if (beta_0 > lb[0]) and (beta_0 < ub[0]):
        beta0_coverage = 1
    if (beta_1 > lb[1]) and (beta_1 < ub[1]):
        beta1_coverage = 1
    if (beta_2 > lb[2]) and (beta_2 < ub[2]):
        beta2_coverage = 1

    return (
        params,
        beta0_coverage, 
        beta1_coverage, 
        beta2_coverage, 
        half_width[0],
        half_width[1],
        half_width[2],
        counts
    )


def main(args):
    methods = args.methods
    N = args.N
    beta_0 = args.beta_0
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    sigma = args.sigma
    n_simulations = args.n_simulations

    data_simulator = DataSimulatorCase1(
        N, beta_0, beta_1, beta_2, sigma, args.scale, args.use_estimated_ps,
        prediction_noise=args.prediction_noise, prediction_bias=args.prediction_bias,
        missing_mechanism=args.missing_mechanism, mispecified_ps_type=args.mispecified_ps_type,
    )

    os.makedirs("results", exist_ok=True)

    for method in methods.split(","):
        mi_coverage_beta0 = 0
        mi_coverage_beta1 = 0
        mi_coverage_beta2 = 0
        
        width_beta0 = []
        width_beta1 = []
        width_beta2 = []
        
        all_params = []

        all_counts = {}

        if args.debug:
            results = []
            for i in range(n_simulations):
                results.append(run_simulation(i, args=args, data_simulator=data_simulator, method=method))
        else:
            # parallel =========
            func = partial(run_simulation, args=args, data_simulator=data_simulator, method=method)
            
            # Use ProcessPoolExecutor to run simulations in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(func, range(n_simulations)))
            # parallel =========

        # Process results returned from parallel execution
        for result in results:
            (
                params, beta0_coverage, beta1_coverage, beta2_coverage,
                width0, width1, width2, counts
                ) = result
            
            mi_coverage_beta0 += beta0_coverage
            mi_coverage_beta1 += beta1_coverage
            mi_coverage_beta2 += beta2_coverage
            width_beta0.append(width0)
            width_beta1.append(width1)
            width_beta2.append(width2)
            all_params.append(params)

            for case, count in sorted(counts.items()):
                if case not in all_counts:
                    all_counts[case] = []
                all_counts[case].append(count)

        print(f"=============== Method: {method} ===============")
        print(", ".join(f"Case {case}: {np.mean(counts_i)}" for case, counts_i in sorted(all_counts.items())))
        print('Coverage of beta_0: ', mi_coverage_beta0 / n_simulations)
        print('Coverage of beta_1: ', mi_coverage_beta1 / n_simulations)
        print('Coverage of beta_2: ', mi_coverage_beta2 / n_simulations)
        print('Average width of beta_0: ', np.mean(width_beta0))
        print('Average width of beta_1: ', np.mean(width_beta1))
        print('Average width of beta_2: ', np.mean(width_beta2))
        print("================================================")
        
        # Analyze estimates bias and variance
        all_params = np.array(all_params)
        bias = np.mean(all_params, axis=0) - np.array([beta_0, beta_1, beta_2])
        variance = np.diag(np.cov(all_params, rowvar=False))
        half_width = 1.96 * np.sqrt(variance)
        print("Simulated Bias: ", bias)
        print("Simulated Variance: ", variance)
        print("Simulated Half width: ", half_width)

        with open(f"results/linear_regression_simulation_{method}_{args.prediction_bias}_{args.prediction_noise}_{args.use_estimated_ps}_{args.missing_mechanism}_{args.mispecified_ps_type}.json", "w") as f:
            f.write(json.dumps({
                "method": method,
                "prediction_bias": args.prediction_bias,
                "prediction_noise": args.prediction_noise,
                "use_estimated_ps": args.use_estimated_ps,
                "mispecified_ps_type": args.mispecified_ps_type,
                "beta0_coverage_rate": mi_coverage_beta0 / n_simulations,
                "beta1_coverage_rate": mi_coverage_beta1 / n_simulations,
                "beta2_coverage_rate": mi_coverage_beta2 / n_simulations,
                "beta0_ci_width": np.mean(width_beta0),
                "beta1_ci_width": np.mean(width_beta1),
                "beta2_ci_width": np.mean(width_beta2),
                "beta0_bias": bias[0],
                "beta1_bias": bias[1],
                "beta2_bias": bias[2],
            }))

if __name__ == "__main__":
    # create argument
    parser = argparse.ArgumentParser(description='Process some integers.')

    # basic parameters (generally should be fixed)
    parser.add_argument("--beta_0", type=float, default=-1, help="beta_0, generally should be fixed")
    parser.add_argument("--beta_1", type=float, default=0.5, help="beta_1, generally should be fixed")
    parser.add_argument("--beta_2", type=float, default=1.2, help="beta_2, generally should be fixed")
    parser.add_argument("--sigma", type=float, default=0.5, help="standard deviation of the noise, generally should be fixed")
    
    # basic parameters (can be changed)
    parser.add_argument("--n_simulations", type=int, default=500, help="number of simulations")

    # parameters for evaluating imputation performance
    parser.add_argument('--methods', type=str, default='WCCA', help='method for imputation')
    parser.add_argument("--N", type=int, default=5000, help="number of labeled data points")
    parser.add_argument("--scale", type=float, default=0.2, help="scale of the data")
    parser.add_argument("--use_estimated_ps", type=bool, default=False, help="Whether to use estimated propensity score")
    parser.add_argument("--mispecified_ps_type", type=int, default=0, help="Whether to use mis-specified propensity score")
    parser.add_argument("--debug", type=bool, default=False)
    
    parser.add_argument("--prediction_noise", type=float, default=0, help="standard deviation of the noise in prediction")
    parser.add_argument("--prediction_bias", type=float, default=0, help="bias in prediction")

    parser.add_argument("--cov_type", type=str, default="HC0")

    parser.add_argument("--missing_mechanism", type=str, default="MAR")
    args = parser.parse_args()

    # print arguments
    print("========================================")
    for arg in vars(args):
        print(arg, getattr(args, arg))
    print("========================================")
    
    main(args)
