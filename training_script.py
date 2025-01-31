import model_functions as mf
from config_script import *
import pickle as pkl
import plotting_functions as pf
all_inputs, all_outputs, all_masks = mf.self_timed_movement_task(config['T_start'], config['T_cue'], config['T_wait'], config['T_movement'], config['T'])

####TRAINING###############
# train on all params
params_nm, losses_nm = mf.fit_nm_rnn(all_inputs, all_outputs, all_masks,
                                  params, optimizer, x0, z0, config['num_full_train_iters'],
                                  config['tau_x'], config['tau_z'], wandb_log=False, modulation=True, noise_std=config['noise_std'])
#save params_nm, which is a dictionary, using pickle
with open('params_nm.pkl', 'wb') as f:
    pkl.dump(params_nm, f)

pf.plot_loss(losses_nm)

all_ys, all_xs, all_zs = mf.test_model(params_nm, noise=True)

pf.plot_output(all_ys)
pf.plot_cue_algn_activity(all_xs, all_zs)
pf.plot_activity_by_area(all_xs, all_zs)


####TRAINING###############
# train on all params
params_nm, losses_nm = mf.fit_nm_rnn(all_inputs, all_outputs, all_masks,
                                  params, optimizer, x0, z0, config['num_full_train_iters'],
                                  config['tau_x'], config['tau_z'], wandb_log=False, modulation=True, noise_std=0)
with open('noiseless_params_nm.pkl', 'wb') as f:
    pkl.dump(params_nm, f)

all_ys, all_xs, all_zs = mf.test_model(params_nm, noise=False)
pf.plot_output(all_ys)
pf.plot_cue_algn_activity(all_xs, all_zs)
pf.plot_activity_by_area(all_xs, all_zs)