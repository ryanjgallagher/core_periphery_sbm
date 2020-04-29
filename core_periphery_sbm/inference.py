import sys
import numpy as np
from scipy.stats import beta
from scipy.special import loggamma

# ------------------------------------------------------------------------------
# ----------------------------- Sampling functions -----------------------------
# ------------------------------------------------------------------------------
def sample_trunc_beta(a, b, lower, upper):
    """
    Samples from a truncated beta distribution in log space

    Parameters
    ----------
    a, b: float
        Canonical parameters of the beta distribution
    lower, upper: float
        Lower and upper truncations of the beta distribution

    Returns
    -------
    s: float
        Sampled value from the truncated beta distribution in log space
    """
    # Check boundaries are correct
    if upper < lower:
        return

    # If a=1 and b=1, then we're sampling truncated uniform distribution
    # (i.e. peak formula below is not valid, but also not needed)
    if a == 1 and b == 1:
        s = np.random.uniform(low=lower, high=upper)
        return s

    # Get location of peak of distribution to determine type of sampling
    peak = (a-1) / (a+b-2)
    # If peak of beta dist is outside truncation, use uniform rejection sampling
    if peak < lower or peak > upper:
        # Sample a proposal
        s = np.random.uniform(low=lower, high=upper)
        # Get components of rejection sampling
        log_f_s = beta.logpdf(s, a, b)
        log_g_s = -1*np.log(upper-lower)
        log_M = max(beta.logpdf(lower,a,b), beta.logpdf(upper,a,b))\
                + np.log(upper-lower)
        # Keep sampling until proposal is accepted
        while np.log(np.random.random()) > log_f_s - (log_M + log_g_s):
            s = np.random.uniform(low=lower, high=upper)
            log_f_s = beta.logpdf(s, a, b)
    # If peak of beta is inside truncation, sample from beta directly
    else:
        s = beta.rvs(a, b)
        # Keep sampling until proposal falls inside truncation boundaries
        while s < lower or s > upper:
            s = beta.rvs(a,b)

    return s

# ------------------------------------------------------------------------------
# ------------------------ Metropolis-Hastings functions -----------------------
# ------------------------------------------------------------------------------
def check_MH_criterion(log_labels_given_ps_prev, log_labels_given_ps_new,
                       log_update, log_revert):
    """
    Checks the Metropolis-Hastings criterion for accepting an MCMC move

    Parameters
    ----------
    log_labels_given_ps_prev: float
        log P(\theta \mid p, A) before the proposed MCMCc move
    log_labels_given_ps_new: float
        log P(\theta \mid p, A) after the proposed move
    log_update: float
        Log probability of transitioning from the current parameter set to the
        proposed parameter set
    log_revert: float
        Log probability of transitioning from the proposed parameter set to the
        current parameter set

    Returns
    -------
    acc: int
        Returns 1 if the move is accepted, 0 otherwsie
    """
    # Calculate posterior ratio
    log_p_update = (log_labels_given_ps_new - log_labels_given_ps_prev)\
                   + (log_revert - log_update)
    log_p_accept = min(0, log_p_update)
    # Decide to accept or reject based on Metropolis-Hastings criterion
    if np.log(np.random.random()) < log_p_accept:
        return 1
    else:
        return 0

# ------------------------------------------------------------------------------
# ------------------------- Model component functions --------------------------
# ------------------------------------------------------------------------------
def get_log_posterior_layered(N_nodes, layer_ms, layer_Ms, layer_ns, layer_ps):
    """
    Calculates the log posterior of the layered core-periphery model

    Parameters
    ----------
    N_nodes: int
        Number of nodes in the network
    layer_ms: 1D array
        Array counting the number of edges that connect to each layer
    layer_Ms: 1D array
        Array counting the maximum number of edges that could potentially
        connect to each layer
    layer_ns: 1D array
        Array counting the number of nodes in each block
    layer_ps: 1D array
        Array recording the density of each layer
    """
    n_layers = len(layer_ps)
    c_likelihood = log_likelihood_layered(layer_ms, layer_Ms, layer_ps)
    c_ps_prior = log_ps_prior_layered(layer_ps)
    c_labels_prior = log_labels_prior_layered(N_nodes, layer_ns, n_layers)

    return c_likelihood + c_ps_prior + c_labels_prior

def get_log_posterior_hubspoke(N_nodes, block_ms, block_Ms, block_ns, block_ps):
    """
    Calculates the log posterior of the hub-and-spoke core-periphery model

    Parameters
    ----------
    N_nodes: int
        Number of nodes in the network
    block_ms: 2D array
        Matrix counting the number of edges between and within each block
    block_Ms: 2D array
        Matrix counting the maximum number of edges that could potentially
        connect between and within each block
    block_ns: 1D array
        Array counting the number of nodes in each block
    block_ps: 2D array
        Matrix recording the density of each block
    """
    c_likelihood = log_likelihood_hubspoke(block_ms, block_Ms, block_ps)
    c_ps_prior = log_ps_prior_hubspoke(block_ps)
    c_labels_prior = log_labels_prior_hubspoke(N_nodes, block_ns)

    return c_likelihood + c_ps_prior + c_labels_prior

# ------------------------------------------------------------------------------
# ---------------------------- Likelihood functions ----------------------------
# ------------------------------------------------------------------------------
def xlogy(x,y):
    if x == 0 and y == 0:
        return 0
    elif y == 0:
        return -1*np.inf
    elif y < 0:
        return
    else:
        return x*np.log(y)

def log_likelihood_layered(layer_ms, layer_Ms, layer_ps):
    """
    Calculates the log likelihood of the layered core-periphery model.

    Parameters
    ----------
    layer_ms: 1D array
        Array counting the number of edges that connect to each layer
    layer_Ms: 1D array
        Array counting the maximum number of edges that could potentially
        connect to each layer
    layer_ps: 1D array
        Array recording the density of each layer
    """
    log_like = 0
    for s in range(len(layer_ps)):
        # Update the layer connection term, p_s^(m_s) * (1-p_s)^(M_s-m_s)
        log_like += xlogy(layer_ms[s], layer_ps[s])
        log_like += xlogy(layer_Ms[s]-layer_ms[s], 1-layer_ps[s])
    return log_like

def log_likelihood_hubspoke(block_ms, block_Ms, block_ps):
    """
    Calculates the log likelihood of the hub-and-spoke core-periphery model.

    Parameters
    ----------
    block_ms: 2D array
        Matrix counting the number of edges between and within each block
    block_Ms: 2D array
        Matrix counting the maximum number of edges that could potentially
        connect between and within each block
    block_ps: 2D array
        Matrix recording the density of each block
    """
    log_like = 0
    for r in range(2):
        for s in range(r+1):
            # Update the layer connection term, p_s^(m_s) * (1-p_s)^(m_s)
            log_like += xlogy(block_ms[r,s], block_ps[r,s])
            log_like += xlogy(block_Ms[r,s]-block_ms[r,s], 1-block_ps[r,s])
    return log_like

# ------------------------------------------------------------------------------
# ---------------------------- Label prior functions ---------------------------
# ------------------------------------------------------------------------------

def log_labels_prior_layered(N_nodes, block_ns, n_layers):
    """
    Calculates the prior on the node labels for the layered modeel

    Parameters
    ----------
    N_nodes: int
        The number of nodes in the network
    block_ns: 1D array
        Array counting the number of nodes in each block
    n_layers: int
        The number of layers being inferred
    """
    c1 = np.sum(loggamma(block_ns + 1))
    c2 = loggamma(N_nodes + 1)
    c3 = loggamma(N_nodes)
    c4 = loggamma(n_layers)
    c5 = loggamma(N_nodes - n_layers - 1)
    c6 = np.log(N_nodes)
    return c1 - c2 - c3 + c4 + c5 - c6

def log_labels_prior_hubspoke(N_nodes, block_ns):
    """
    Calculates the prior on the node labels for the hub-and-spoke model

    Parameters
    ----------
    N_nodes: int
        The number of nodes in the network
    block_ns: 1D array
        Array counting the number of nodes in each block
    """
    c1 = np.sum(loggamma(block_ns + 1))
    c2 = loggamma(N_nodes + 1)
    c3 = loggamma(N_nodes)
    c4 = loggamma(2)
    c5 = loggamma(N_nodes - 2 - 1)
    c6 = np.log(N_nodes)
    return c1 - c2 - c3 + c4 + c5 - c6

# ------------------------------------------------------------------------------
# ----------------------------- ps prior functions -----------------------------
# ------------------------------------------------------------------------------
def log_ps_prior_layered(layer_ps):
    """
    Calculates the prior on the ps for the layered model

    Parameters
    ----------
    layer_ps: 1D array
        Array recording the density of each layer
    """
    # Check if ps are ordered
    if np.all(layer_ps[:-1] >= layer_ps[1:]):
        return loggamma(len(layer_ps))
    else:
        # Violates the ordering criterion
        return -1*np.inf

def log_ps_prior_hubspoke(block_ps):
    """
    Calculates the prior on the ps for the hub-and-spoke model

    Parameters
    ----------
    block_ps: 2D array
        Matrix recording the density of each block
    """
    if block_ps[0,0] >= block_ps[0,1] and block_ps[0,1] >= block_ps[1,1]:
        return np.log(6)
    else:
        # Violates the ordering criterion
        return -1*np.inf

# ------------------------------------------------------------------------------
# ------------------------ Theta given p and A functions -----------------------
# ------------------------------------------------------------------------------
def log_labels_given_ps_layered(N_nodes, block_ns, layer_ms, layer_Ms, layer_ps):
    """
    Calculates P(\theta \mid A, p) for the layered model

    Parameters
    ----------
    N_nodes: int
        Number of nodes in the network
    block_ns: 1D array
        Array counting the number of nodes in each block
    layer_ms: 1D array
        Array counting the number of edges that connect to each layer
    layer_Ms: 1D array
        Array counting the maximum number of edges that could potentially
        connect to each layer
    layer_ps: 1D array
        Array recording the density of each layer
    """
    n_layers = len(layer_ps)
    log_like = log_likelihood_layered(layer_ms, layer_Ms, layer_ps)
    log_label_prior = log_labels_prior_layered(N_nodes, block_ns, n_layers)
    return log_like + log_label_prior


def log_labels_given_ps_hubspoke(N_nodes, block_ns, block_ms, block_Ms, block_ps):
    """
    Calculates P(\theta \mid A, p) for the hub-and-spoke model

    Parameters
    ----------
    N_nodes: int
        Number of nodes in the network
    block_ns: 1D array
        Array counting the number of nodes in each block
    block_ms: 2D array
        Matrix counting the number of edges between and within each block
    block_Ms: 2D array
        Matrix counting the maximum number of edges that could potentially
        connect between and within each block
    block_ps: 2D array
        Matrix recording the density of each block
    """
    log_like = log_likelihood_hubspoke(block_ms, block_Ms, block_ps)
    log_label_prior = log_labels_prior_hubspoke(N_nodes, block_ns)
    return log_like + log_label_prior
