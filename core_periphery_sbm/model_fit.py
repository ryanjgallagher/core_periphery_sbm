import numpy as np
import networkx as nx
import inference as infh
import network_helper as nh
from scipy.stats import dirichlet

# ------------------------------------------------------------------------------
# ----------------------------- Sampling functions -----------------------------
# ------------------------------------------------------------------------------
def sample_recursive_ps_layered(n_layers, n_samples=1):
    """
    Samples from the recursive distribution for the ps of the layered model

    Returns
    -------
    ps: 2D array
        Array of array of ps drawn from the recursive ps distribution
    """
    ps = np.zeros((n_samples,n_layers))
    for sample_n in range(n_samples):
        for layer_n in range(n_layers):
            if layer_n == 0:
                ps[sample_n,layer_n] = np.random.uniform(high=1)
            else:
                ps[sample_n,layer_n] = np.random.uniform(high=ps[sample_n,layer_n-1])
    return ps

def sample_dirichlet_ps_prior_layered(n_layers, n_samples=1):
    """
    Samples from the Dirichlet ps prior for the layered model

    Returns
    -------
    ps: 2D array
        Array of arrays of ps drawn from the Dirichlet ps prior
    """
    spacings = dirichlet.rvs([1]*(n_layers+1), size=n_samples)
    cuml_spacings = np.cumsum(spacings, axis=1)
    sampled_ps = (1 - cuml_spacings)[:,:n_layers] # only take l out of l+1 spacings
    return sampled_ps

def sample_recursive_ps_hubspoke(n_samples=1):
    """
    Samples from the recursive ps distribution of the hub-and-spoke model

    Returns
    -------
    ps: 3D array
        Array of 2D arrays of ps drawn from recursive ps distribution
    """
    ps = np.zeros((n_samples,2,2))
    for sample_n in range(n_samples):
        for r in range(2):
            for s in range(r+1):
                if r == 0 and s == 0:
                    ps[sample_n,0,0] = np.random.uniform(high=1)
                elif (r == 0 and s == 1) or (r == 1 and s == 0):
                    p_01 = np.random.uniform(high=ps[sample_n,0,0])
                    ps[sample_n,0,1] = p_01
                    ps[sample_n,1,0] = p_01
                else:
                    ps[sample_n,1,1] = np.random.uniform(high=ps[sample_n,0,1])
    return ps

def sample_dirichlet_ps_prior_hubspoke(n_samples=1):
    """
    Samples from the Dirichlet ps prior for the hub-and-spoke model

    Returns
    -------
    ps: 3D array
        Array of 2D arrays of ps drawn from Dirichlet ps prior
    """
    spacings = dirichlet.rvs([1, 1, 1, 1], size=n_samples)
    cuml_spacings = np.cumsum(spacings, axis=1)
    sampled_ps = np.zeros((n_samples, 2, 2))
    sampled_ps[:,0,0] = 1 - cuml_spacings[:,0]
    sampled_ps[:,0,1] = 1 - cuml_spacings[:,1]
    sampled_ps[:,1,0] = 1 - cuml_spacings[:,1]
    sampled_ps[:,1,1] = 1 - cuml_spacings[:,2]
    return sampled_ps

# ------------------------------------------------------------------------------
# ---------------------------- Likelihood functions ----------------------------
# ------------------------------------------------------------------------------
def log_ps_recursive_likelihood_layered(ps):
    """
    Calculates the likelihood of the recursive ps dist for the layered model

    Parameters
    ----------
    ps: 1D array
        layer ps
    """
    n_layers = len(ps)
    return np.sum((-1 * np.log(ps))[:(n_layers-1)])

def log_ps_recursive_likelihood_hubspoke(ps):
    """
    Calculates the likelihood of the recursive ps dist for the hub-and-spoke model
    """
    return -1 * np.log(ps[0,0]) - np.log(ps[0,1])

# ------------------------------------------------------------------------------
# --------------------- Minimum description length functions -------------------
# ------------------------------------------------------------------------------
def mdl_from_samples(log_likes, log_weights, log_labels_prior):
    """
    Approximates the minimum description length of a core-periphery model
    partition given samples

    Parameters
    ----------
    log_likes: 1D array
        Values of the log likelihood of a core-periphery model evaluated for a
        fixed partition
    log_weights: 1D array
        Weights of each log likelihood according to the importance sampling
        schema
    log_labels_prior: float
        log P(\theta) for the fixed partition

    Returns
    -------

    """
    n_samples = log_likes.shape[0]
    weighted_log_likes = log_likes + log_weights
    max_log_like = max(weighted_log_likes)
    delta_log_likes = weighted_log_likes - max_log_like

    np.seterr(under='ignore')
    mdl = max_log_like + log_labels_prior - np.log(n_samples) \
          + np.log(np.sum(np.exp(delta_log_likes)))
    np.seterr(under='raise')

    return -1 * mdl

def mdl_layered(G, node_labels, n_layers, n_samples=10000, details=False, relabel=True):
    """
    Approximates the minimum description length of a partition found through the
    layered model. Assumes no self loops in network.

    Parameters
    ----------
    G: NetworkX graph
        Network for which the core-periphery structure was inferred
    node_labels: 1D array
        Array of the block labels for each node
    n_layers: int
        The number of layers in the core-periphery model
    n_samples: int
        The number of samples of the ps prior to take in evaluating the log
        likelihood
    details:
        If true, returns all of the sampled ps and log likelihood values
    relabel:
        If true, relabel graph to 0-indexed integer labels. Only use False if
        graph is already indexed that way
    """
    if relabel:
        H = nx.convert_node_labels_to_integers(G, ordering='sorted')
    else:
        H = G.copy()
    # Get the number of nodes in each block and the inter-block edge counts
    block_ns,block_ms,block_Ms = nh.get_block_stats(H, node_labels,n_blocks=n_layers)
    layer_ms,layer_Ms = nh.get_layered_stats(block_ns, block_ms)
    N_nodes = np.sum(block_ns)
    # Sample ps from prior and evaluate the likelihood for each sample of ps
    sampled_ps = sample_recursive_ps_layered(n_layers, n_samples=n_samples)
    log_likes = np.array([infh.log_likelihood_layered(layer_ms, layer_Ms, ps)
                          for ps in sampled_ps])
    log_weights = np.array([infh.log_ps_prior_layered(ps)
                            - log_ps_recursive_likelihood_layered(ps)
                            for ps in sampled_ps])
    # Get the contribution of the prior on the labels
    log_labels_prior = infh.log_labels_prior_layered(N_nodes, block_ns, n_layers)
    # Calculate the minimum description length from the distribution samples
    mdl = mdl_from_samples(log_likes, log_weights, log_labels_prior)

    if details:
        return mdl,sampled_ps,log_likes,log_weights
    else:
        return mdl

def mdl_hubspoke(G, node_labels, n_samples=10000, details=False, relabel=True):
    """
    Approximates the minimum description length of a partition found through the
    hub-and-spoke model. Assumes no self loops in network.

    Parameters
    ----------
    G: NetworkX graph
        Network for which the core-periphery structure was inferred
    node_labels: 1D array
        Array of the block labels for each node
    n_samples: int
        The number of samples of the ps prior to take in evaluating the log
        likelihood
    details:
        If true, returns all of the sampled ps and log likelihood values
    relabel:
        If true, relabel graph to 0-indexed integer labels. Only use False if
        graph is already indexed that way
    """
    if relabel:
        H = nx.convert_node_labels_to_integers(G, ordering='sorted')
    else:
        H = G.copy()
    # Get the number of nodes in each block and the inter-block edge counts
    block_ns,block_ms,block_Ms = nh.get_block_stats(H, node_labels, n_blocks=2)
    N_nodes = np.sum(block_ns)
    # Sample ps from prior and evaluate the likelihood for each sample of ps
    sampled_ps = sample_recursive_ps_hubspoke(n_samples=n_samples)
    log_likes = np.array([infh.log_likelihood_hubspoke(block_ms, block_Ms, ps)
                          for ps in sampled_ps])
    log_weights = np.array([infh.log_ps_prior_hubspoke(ps)
                            - log_ps_recursive_likelihood_hubspoke(ps)
                            for ps in sampled_ps])
    # Get the contribution of the prior on the labels
    log_labels_prior = infh.log_labels_prior_hubspoke(N_nodes, block_ns)
    # Calculate the minimum description length from the distribution samples
    mdl = mdl_from_samples(log_likes, log_weights, log_labels_prior)

    if details:
        return mdl,sampled_ps,log_likes,log_weights
    else:
        return mdl
