from jax import random as jr
from jax import numpy as jnp
import optax
import math


def init_params(key, n_bg, n_nm, g_bg, g_nm, input_dim, output_dim):
    # for now assume Th/BG/C are same size, g is the same for all weight matrices
    skeys = jr.split(key, 17)

    # bg parameters
    J_bg = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[0], (n_bg, n_bg))
    B_bgc = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[1], (n_bg, n_bg))

    # c parameters
    J_c = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[2], (n_bg, n_bg))
    B_cu = (1 / math.sqrt(input_dim)) * jr.normal(skeys[3], (n_bg, input_dim))
    B_ct = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[4], (n_bg, n_bg))

    # t parameters
    J_t = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[5], (n_bg, n_bg))
    B_tbg = (g_bg / math.sqrt(n_bg)) * jr.normal(skeys[6], (n_bg, n_bg))

    # nm parameters
    J_nm = (g_nm / math.sqrt(n_nm)) * jr.normal(skeys[7], (n_nm, n_nm))
    J_nmc = (g_nm / math.sqrt(n_nm)) * jr.normal(skeys[8], (n_nm, n_bg))
    B_nmc = (1 / math.sqrt(n_nm)) * jr.normal(skeys[9], (n_nm, n_bg))

    m = (1 / math.sqrt(n_nm)) * jr.normal(skeys[10], (1, n_nm))
    c = (1 / math.sqrt(n_nm)) * jr.normal(skeys[11])

    U = (1 / math.sqrt(n_bg)) * jr.normal(skeys[12], (1, n_bg))
    V_bg = (1 / math.sqrt(n_bg)) * jr.normal(skeys[13], (1, n_bg))
    V_c = (1 / math.sqrt(n_bg)) * jr.normal(skeys[14], (1, n_bg))

    # readout params
    C = (1 / math.sqrt(n_bg)) * jr.normal(skeys[15], (output_dim, n_bg))
    rb = (1 / math.sqrt(n_bg)) * jr.normal(skeys[16], (output_dim, ))

    return {
        'J_bg': J_bg,
        'B_bgc': B_bgc,
        'J_c': J_c,
        'B_cu': B_cu,
        'B_ct': B_ct,
        'J_t': J_t,
        'B_tbg': B_tbg,
        'J_nm': J_nm,
        'J_nmc': J_nmc,
        'B_nmc': B_nmc,
        'm': m,
        'c': c,
        'C': C,
        'rb': rb,
        'U': U,
        'V_bg': V_bg,
        'V_c': V_c
    }

#generate a key
key = jr.PRNGKey(13)
# parameters we want to track in wandb
default_config = dict(
    # model parameters
    n_bg=20,
    n_nm=5,      # NM (SNc) dimension
    g_bg=1.4,
    g_nm=1.4,
    U=1,      # input dim
    O=1,      # output dimension
    # Model Hyperparameters
    tau_x=10,
    tau_z=100,
    noise_std=0.1,  # Standard deviation of noise
    # Timing (task) parameters
    dt=10, # ms
    # Data Generation
    #T_start = jnp.arange(300,405,5),
    #choose 200 random numbers between 100 and 400 for T_start
    T_start=jr.randint(key, shape=(200,), minval=100, maxval=400),
    T_cue=10,
    T_wait = 300,
    T_movement=300, #three second window to move
    T=900,
    # Training
    num_nm_only_iters=0,
    num_full_train_iters=10000,
    keyind=13,
)

# declare the config
config = default_config

# set up the random key
key = jr.PRNGKey(config['keyind'])

# initialize the parameters
params = init_params(
    key,
    config['n_bg'], config['n_nm'],
    config['g_bg'], config['g_nm'],
    config['U'], config['O']
)

#set up the number of cells in the BG and NM
n_d1_cells = config['n_bg'] // 2
n_d2_cells = config['n_bg'] - n_d1_cells

#set up the optimizer
optimizer = optax.chain(
  optax.clip(1.0), # gradient clipping
  optax.adamw(learning_rate=1e-3),
)

x_bg0 = jnp.ones((config['n_bg'],)) * 0
x_c0 = jnp.ones((config['n_bg'],)) * 0
x_t0 = jnp.ones((config['n_bg'],)) * 0
x0 = (x_bg0, x_c0, x_t0)
z0 = jnp.ones((config['n_nm'],)) * 0

#declare testing params
n_seeds = 50
test_noise_std = 0.15  # Specify noise standard deviation for testing
#test_start_t = jnp.arange(300,405,5)
test_start_t = jnp.arange(275, 330, 5)
#opto params
n_opto_seeds = 300
opto_tstart = 300 #start of cue for opto experiments
opto_start = opto_tstart + 100#start of opto stimulation
opto_end = opto_start + 175 #end of opto stimulation
d1_stim_strength = 0.3
d2_stim_strength = 0.2

control_stim = jnp.zeros((config['n_bg'],))
suppress_d1_stim = jnp.array([-d1_stim_strength] * n_d1_cells + [0] * n_d2_cells)
suppress_d2_stim = jnp.array([0] * n_d1_cells + [-d2_stim_strength] * n_d2_cells)
enhance_d1_stim = jnp.array([d1_stim_strength] * n_d1_cells + [0] * n_d2_cells)
enhance_d2_stim = jnp.array([0] * n_d1_cells + [d2_stim_strength] * n_d2_cells)

spatial_stim_list = [control_stim, suppress_d1_stim, suppress_d2_stim, enhance_d1_stim,
                     enhance_d2_stim]
