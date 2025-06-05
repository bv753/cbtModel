import model_functions as mf
import plotting_functions as pf
import pickle as pkl

#load params_nm
with open('params_nm.pkl', 'rb') as f:
    params_nm = pkl.load(f)

opto_ys, opto_xs, opto_zs = mf.simulate_opto(params_nm)
#pf.plot_opto_inh(opto_ys, opto_xs, opto_zs)
#pf.plot_opto_stim(opto_ys, opto_xs, opto_zs)
pf.plot_opto(opto_xs, opto_zs)