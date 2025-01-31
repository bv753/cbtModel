from model_functions import *
from config_script import *
import pickle as pkl
import plotting_functions as pf
all_inputs, all_outputs, all_masks = self_timed_movement_task(config['T_start'], config['T_cue'], config['T_wait'], config['T_movement'], config['T'])

####TRAINING###############
# train on all params
params_nm, losses_nm = fit_nm_rnn(all_inputs, all_outputs, all_masks,
                                  params, optimizer, x0, z0, config['num_full_train_iters'],
                                  config['tau_x'], config['tau_z'], wandb_log=False, modulation=True)
#save params_nm, which is a dictionary, using pickle
with open('params_nm.pkl', 'wb') as f:
    pkl.dump(params_nm, f)

pf.plot_loss(losses_nm)
