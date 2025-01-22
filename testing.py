import model_functions as mf
import plotting_functions as pf
import pickle as pkl

#load params_nm
with open('params_nm.pkl', 'rb') as f:
    params_nm = pkl.load(f)

####TESTING###############
all_ys, all_xs, all_zs = mf.test_model(params_nm)

C = params_nm['C']