import numpy as np

def dynamic_pop_size_UMDA(n_items, noise):
    return int(20 * np.sqrt(n_items) * np.log(n_items))

def dynamic_pop_size_PCEA(n_items, noise):
    return int(10 * np.sqrt(n_items) * np.log(n_items))

def dynamic_pop_size_mu(n_items, noise):
    return int(max(noise * noise, 1) * np.log(n_items))

def inverse_n_mut_rate(n_items, noise):
    return 1/n_items