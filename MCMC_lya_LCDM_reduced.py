import numpy as np
import emcee
from scipy.interpolate import interp1d
from scipy.linalg import block_diag
import pprint, pickle, os
from datetime import date

###################### USER INPUT #################################################

###choose data-set
#DATASET = "mike-hires"; zp = 4.5
DATASET = "xq-100"; zp = 3.6
#DATASET = "joint"; zp = 4.2

### choose prior on zreio, sigma8, neff
# cosmo_prior = "flat"
# cosmo_prior = "ede"
# cosmo_prior = "lcdm"
cosmo_prior = "lya"
# cosmo_prior = "s8_0p2"

### choose smooth power-law IGM thermal history or T(z)-free
igm_history = "Tpowerlaw"
#igm_history = "Tfree"

###########################################################################
##priors on zreio, sigma8, neff
if cosmo_prior == "ede":
    mu_cosmo = [7.5, 0.830]; sigma_cosmo = [1., 0.015]
    log_pr_cosmo = np.zeros(len(mu_cosmo))
elif cosmo_prior == "lcdm":
    mu_cosmo = [7.5, 0.830]; sigma_cosmo = [1., 0.015]
    log_pr_cosmo = np.zeros(len(mu_cosmo))
elif cosmo_prior == "lya":
    mu_cosmo = [7.5, 0.830]; sigma_cosmo = [1., 0.015]; ll_neff = -2.43; ul_neff = -2.32
    log_pr_cosmo = np.zeros(len(mu_cosmo))
elif cosmo_prior == "s8_0p2":
    mu_cosmo = [7.5, 0.830]; sigma_cosmo = [1., 0.2]; ll_neff = -2.43; ul_neff = -2.32
    log_pr_cosmo = np.zeros(len(mu_cosmo))

gr_folder = "./grids_and_data/"
NOMEFILE = "chain_"+str(date.today())+"_"+igm_history

            ########################################### ASTRO PARAMS INPUTs ###########################################
#### REDSHIFT INDEPENDENT PARAMS - params order: z_reio, sigma_8, n_eff, f_UV
other_param_size = [3, 5, 5, 3] #how many values I have for each param
other_param_min = np.array([7., 0.754, -2.3474, 0.])
other_param_max = np.array([15., 0.904, -2.2674, 1.])
other_param_ref = np.array([9., 0.829, -2.3074, 0.])
zreio_range = other_param_max[0]-other_param_min[0]; neff_range = other_param_max[2]-other_param_min[2]

#### REDSHIFT DEPENDENT PARAMS - params order: params order: mean_f , t0, slope
zdep_params_size = [9, 3, 3] #how many values I have for each param
zdep_params_refpos = [4, 1, 2] #where to store the P_F(ref)

####MEAN FLUXES values###
flux_ref_old = (np.array([0.669181, 0.617042, 0.564612, 0.512514, 0.461362, 0.411733, 0.364155, 0.253828, 0.146033, 0.0712724]))

######################################################## DATA #################################################################################
###### FIRST DATASET (19 wavenumbers) #####################################    ***XQ-100***
zeta_range_19 = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2]  #list of redshifts corresponding to the 19 wavenumbers (k) (see ./SPECTRA/README)
k_19 = [0.003,0.006,0.009,0.012,0.015,0.018,0.021,0.024,0.027,0.03,0.033,0.036,0.039,0.042,0.045,0.048,0.051,0.054,0.057]

###### SECOND DATASET (7 wavenumbers) #####################################    ***HIRES/MIKE***
zeta_range_7 = [4.2, 4.6, 5.0, 5.4]  #list of redshifts corresponding to the 7 wavenumbers (k) (see ./SPECTRA/README)
k_7 = [0.00501187,0.00794328,0.0125893,0.0199526,0.0316228,0.0501187,0.0794328]

zeta_full_lenght = (len(zeta_range_19) + len(zeta_range_7))
kappa_full_lenght = (len(k_19) + len(k_7))
redshift = [3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.2, 4.6, 5.0, 5.4] #which snapshots (first 7 for first dataset, last 4 for second one)

########T0 AND SLOPE VALUES##########
t0_ref_old = np.array([11251.5, 11293.6, 11229.0, 10944.6, 10421.8, 9934.49, 9227.31, 8270.68, 7890.68, 7959.4])
slope_ref_old = np.array([1.53919, 1.52894, 1.51756, 1.50382, 1.48922, 1.47706, 1.46909, 1.48025, 1.50814, 1.52578])

t0_values_old = np.zeros(( 10, zdep_params_size[1] ))
t0_values_old[:,0] = np.array([7522.4, 7512.0, 7428.1, 7193.32, 6815.25, 6480.96, 6029.94, 5501.17, 5343.59, 5423.34])
t0_values_old[:,1] = t0_ref_old[:]
t0_values_old[:,2] = np.array([14990.1, 15089.6, 15063.4, 14759.3, 14136.3, 13526.2, 12581.2, 11164.9, 10479.4, 10462.6])

slope_values_old = np.zeros(( 10, zdep_params_size[2] ))
slope_values_old[:,0] = np.array([0.996715, 0.979594, 0.960804, 0.938975, 0.915208, 0.89345, 0.877893, 0.8884, 0.937664, 0.970259])
slope_values_old[:,1] = [1.32706, 1.31447, 1.30014, 1.28335, 1.26545, 1.24965, 1.2392, 1.25092, 1.28657, 1.30854]
slope_values_old[:,2] = slope_ref_old[:]

t0_min = t0_values_old[:,0]*0.1; t0_max = t0_values_old[:,2]*1.4
slope_min = slope_values_old[:,0]*0.8; slope_max = slope_values_old[:,2]*1.15

######################### IMPORTING THE TWO GRIDS FOR KRIGING #########################################
#### Here I import the gridS that I pre-computed  

#### HIGH REDSHIFT
pkl = open(gr_folder+'full_matrix_interpolated_ASTRO_reduced.pkl', 'r')
full_matrix_interpolated_ASTRO = pickle.load(pkl)
grid_lenght_ASTRO = len(full_matrix_interpolated_ASTRO[0,0,:])
print full_matrix_interpolated_ASTRO.shape

#### LOW REDSHIFT
pkl = open(gr_folder+'full_matrix_interpolated_ASTRO_LR_reduced.pkl', 'r')
full_matrix_interpolated_ASTRO_LR = pickle.load(pkl)
grid_lenght_ASTRO_LR = len(full_matrix_interpolated_ASTRO_LR[0,0,:])


ALL_zdep_params = len(flux_ref_old) + len(t0_ref_old) + len(slope_ref_old)
astroparams_number_KRIG = len(other_param_size) + ALL_zdep_params
print full_matrix_interpolated_ASTRO_LR.shape

#### --- ASTRO GRID --- ORDER OF THE COMPLETE LIST OF ASTRO PARAMS: z_reio, sigma_8, n_eff, f_UV, mean_f(z), t0(z), slope(z)

#### HIGH REDSHIFT
X = np.zeros(( grid_lenght_ASTRO, astroparams_number_KRIG ))
for param_index in range(astroparams_number_KRIG):
    X[:,param_index] = np.genfromtxt(gr_folder+'kriging_GRID_2R_astro_18p_HR_noPRACE_reduced.dat', usecols=[param_index], skip_header=1)
print X.shape

#### LOW REDSHIFT
X_LR = np.zeros(( grid_lenght_ASTRO_LR, astroparams_number_KRIG ))
for param_index in range(astroparams_number_KRIG):
    X_LR[:,param_index] = np.genfromtxt(gr_folder+'kriging_GRID_2R_astro_LR_noPRACE_reduced.dat', usecols=[param_index], skip_header=1)
print X_LR.shape

##################################### FUNCTIONS FOR THE ORDINARY KRIGING INTERPOLATION ON ALL THE (k,z) COMBINATIONS #################################
epsilon = 1e-8
exponent = 6.
###########################  STUFF FOR INTERPOLATING IN THE ASTROPARAMS SPACE ####################################
redshift_list = np.array([3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.6, 5.0, 5.4]) #combined dataset (MIKE/HIRES + XQ-100)
astrokrig_result = np.zeros(( zeta_full_lenght, kappa_full_lenght ))
###minimum and maximum values for the kriging normalisation###
F_prior_min = np.array([0.535345,0.493634,0.44921,0.392273,0.338578,0.28871,0.218493,0.146675,0.0676442,0.0247793])
F_prior_max = np.array([0.803017,0.748495,0.709659,0.669613,0.628673,0.587177,0.545471,0.439262,0.315261,0.204999])


############## KRIGING INTERPOLATION

############### these 4 functions are universal because the interpolation is done independently for each (k,z)-bin
def z_dep_func(parA, parS, z):  #analytical function for the redshift dependence of t0 and slope
    return parA*(( (1.+z)/(1.+zp) )**parS)

def ordkrig_distance(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7):
    return (((p1 - v1)**2 + (p2 - v2)**2 + (p3 - v3)**2 + (p4 - v4)**2 + (p5 - v5)**2 + (p6 - v6)**2 + (p7 - v7)**2)**(0.5) + epsilon)**exponent

def ordkrig_norm(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7):
    return np.sum(1./ordkrig_distance(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7))

def ordkrig_lambda(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7):
    return (1./ordkrig_distance(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7))/ordkrig_norm(p1, p2, p3, p4, p5, p6, p7, v1, v2, v3, v4, v5, v6, v7)
##################

if igm_history == "Tpowerlaw" and DATASET == "mike-hires":
    ndim = 12; iflux = 6        
    n = ndim - 4;  a = iflux+1
    
elif igm_history == "Tpowerlaw" and DATASET != "mike-hires":
    ndim = 18; iflux = 0; n = ndim - 4

elif igm_history == "Tfree" and DATASET == "mike-hires":
    ndim = 14; iflux = 6
    n = ndim - 4;  a = iflux+1

elif igm_history == "Tfree" and DATASET != "mike-hires":
    ndim = 26; iflux = 0; n = ndim - 4

def ordkrig_estimator(p21, z):
    pb10 = z_dep_func(p21[-2], p21[-1], z[:])/(slope_max[:]-slope_min[:])
    
    if igm_history == "Tpowerlaw":
        pa10 = (z_dep_func(p21[n], p21[n+1], z[:])*1e4)/(t0_max[:]-t0_min[:])
    elif igm_history == "Tfree" and DATASET == "mike-hires":
        pa10 = np.array([p21[8], p21[9], p21[10], p21[11]])*1e4/(t0_max[6:]-t0_min[6:])
    elif igm_history == "Tfree" and DATASET != "mike-hires":
        pa10 = np.array([p21[14], p21[15], p21[16], p21[17],p21[18], p21[19], p21[20], p21[21], p21[22], p21[23]])*1e4/(t0_max[:]-t0_min[:])
    
    p37 = np.concatenate((p21[:-2], pa10[iflux:], pb10[iflux:]))
    
    if DATASET == "mike-hires":
        for index in range(a,len(redshift)):
            astrokrig_result[index,:] = np.sum(np.multiply(ordkrig_lambda(p37[0]/zreio_range, p37[1], p37[2]/neff_range, p37[3], p37[4+index-a]/(F_prior_max[index-1]-F_prior_min[index-1]), p37[8+index-a], p37[12+index-a], X[:,0], X[:,1], X[:,2], X[:,3], X[:,4+index-1], X[:,14+index-1], X[:,24+index-1]), full_matrix_interpolated_ASTRO[index,:,:]),axis=1)

    elif DATASET != "mike-hires":
        for index in range(iflux,len(redshift)):
            if index < iflux: #low redshift interpolation (9+4 flux points)   [3.0 <= z <= 4.0]
                    astrokrig_result[index,:] = np.sum(np.multiply(ordkrig_lambda(p37[0], p37[1], p37[2], p37[3], (p37[4+index]/(F_prior_max[index]-F_prior_min[index])), p37[14+index], p37[24+index], X_LR[:,0], X_LR[:,1], X_LR[:,2], X_LR[:,3], X_LR[:,4+index], X_LR[:,14+index], X_LR[:,24+index]), full_matrix_interpolated_ASTRO_LR[index,:,:]),axis=1)
            elif index == iflux: #superposed redshift interpolation (18 flux points)   [z = 4.2]
                    astrokrig_result[index,:] = np.sum(np.multiply(ordkrig_lambda(p37[0], p37[1], p37[2], p37[3], (p37[4+index]/(F_prior_max[index]-F_prior_min[index])), p37[14+index], p37[24+index], X[:,0], X[:,1], X[:,2], X[:,3], X[:,4+index], X[:,14+index], X[:,24+index]), full_matrix_interpolated_ASTRO[index,:,:]),axis=1)
            elif index > iflux: #high redshift interpolation (18 flux points)          [AGAIN z=4.2 PLUS 4.6 <= z <= 5.4]
                    astrokrig_result[index,:] = np.sum(np.multiply(ordkrig_lambda(p37[0]/zreio_range, p37[1], p37[2]/neff_range, p37[3], (p37[4+index-1]/(F_prior_max[index-1]-F_prior_min[index-1])), p37[14+index-1], p37[24+index-1], X[:,0], X[:,1], X[:,2], X[:,3], X[:,4+index-1], X[:,14+index-1], X[:,24+index-1]), full_matrix_interpolated_ASTRO[index,:,:]),axis=1)
    
    # print len(p21), len(p37), pa10, pb10
    return astrokrig_result    
    
#################################################################################################################################################

####################################################### FUNCTIONS FOR emcee ######################################################
model_H = np.zeros (( len(zeta_range_7), len(k_7) )); y_H = np.zeros (( len(zeta_range_7), len(k_7) )) 
model_M = np.zeros (( len(zeta_range_7)-1, len(k_7) )); y_M = np.zeros (( len(zeta_range_7)-1, len(k_7) ))
model_XQ = np.zeros (( len(zeta_range_19), len(k_19) )); y_XQ = np.zeros (( len(zeta_range_19), len(k_19) )) 

pkl = open(gr_folder+'y_M_reshaped.pkl', 'r')
y_M_reshaped = pickle.load(pkl)

pkl = open(gr_folder+'y_H_reshaped.pkl', 'r')
y_H_reshaped = pickle.load(pkl)

pkl = open(gr_folder+'cov_H_inverted.pkl', 'r')
cov_H_inverted = pickle.load(pkl)

pkl = open(gr_folder+'cov_M_inverted.pkl', 'r')
cov_M_inverted = pickle.load(pkl)

pkl = open(gr_folder+'PF_noPRACE.pkl', 'r')
PF_noPRACE = pickle.load(pkl)

pkl = open(gr_folder+'y_XQ_reshaped.pkl', 'r')
y_XQ_reshaped = pickle.load(pkl)

pkl = open(gr_folder+'cov_XQ_inverted.pkl', 'r')
cov_XQ_inverted = pickle.load(pkl)

pkl = open(gr_folder+'cov_COMBINED_inverted.pkl', 'r')
cov_COMBINED_inverted = pickle.load(pkl)

pkl = open(gr_folder+'y_COMBINED_reshaped.pkl', 'r')
y_COMBINED_reshaped = pickle.load(pkl)

cov_MH_inverted = block_diag(cov_H_inverted,cov_M_inverted)
y_MH_reshaped = np.concatenate((y_H_reshaped, y_M_reshaped))
    
#### CHI^2 (likelihood)
def lnlike(theta, z, DATASET):
    model = PF_noPRACE*ordkrig_estimator(theta, z)
    upper_block = np.vsplit(model, [7,11])[0]; lower_block = np.vsplit(model, [7,11])[1]
    if DATASET == "mike-hires":
        model_H[:,:] = lower_block[:,19:]; model_H_reshaped = np.reshape(model_H, -1, order='C')
        model_M[:,:] = lower_block[:3,19:]; model_M_reshaped = np.reshape(model_M, -1, order='C')
        model_MH_reshaped = np.concatenate((model_H_reshaped,model_M_reshaped))
        chi2 = np.dot((y_MH_reshaped - model_MH_reshaped),np.dot(cov_MH_inverted,(y_MH_reshaped - model_MH_reshaped)))
    elif DATASET == "xq-100":
        model_XQ[:,:] = upper_block[:,:19]; model_XQ_reshaped = np.reshape(model_XQ, -1, order='C')
        chi2 = np.dot((y_XQ_reshaped - model_XQ_reshaped),np.dot(cov_XQ_inverted,(y_XQ_reshaped - model_XQ_reshaped)))
    elif DATASET == "joint":
        model_XQ[:,:] = upper_block[:,:19]; model_XQ_reshaped = np.reshape(model_XQ, -1, order='C')
        model_H[:,:] = lower_block[:,19:]; model_H_reshaped = np.reshape(model_H, -1, order='C')
        model_M[:,:] = lower_block[:3,19:]; model_M_reshaped = np.reshape(model_M, -1, order='C')
        model_COMBINED_reshaped = np.concatenate((model_XQ_reshaped,model_H_reshaped,model_M_reshaped))
        chi2 = np.dot((y_COMBINED_reshaped - model_COMBINED_reshaped),np.dot(cov_COMBINED_inverted,(y_COMBINED_reshaped - model_COMBINED_reshaped)))
    return -0.5 * (chi2)

#########################FOR THE MCMC##########################################
if DATASET == "mike-hires":
    mu = flux_ref_old[iflux:]
else:
    mu = flux_ref_old

sigma = 0.04; log_pr = np.zeros(len(mu))

if igm_history == "Tpowerlaw":
    
    def lnprior(theta):
        g = z_dep_func(theta[-2],theta[-1],redshift_list[iflux:])
        if 1. <= g[0] <= 1.7 and 1. <= g[1] <= 1.7 and 1. <= g[2] <= 1.7 and 1. <= g[3] <= 1.7:
            if 6. <= theta[0] <= other_param_max[0] and 0.5 <= theta[1] <= 1.5 and ll_neff <= theta[2] <= ul_neff and 0. <= theta[3] <= 1. and 0.5 <= theta[n] <= 1.5 and -5. <= theta[n+1] <= 5.:
            
                log_pr[:] = -0.5*(theta[4:n] - mu[:])**2/sigma**2
                if cosmo_prior == "flat":
                    return np.sum(log_pr) 
                elif cosmo_prior == "ede" or cosmo_prior == "lcdm" or cosmo_prior == "lya":
                    for i in range(len(mu_cosmo)):
                        log_pr_cosmo[i] = -0.5*(theta[i] - mu_cosmo[i])**2/sigma_cosmo[i]**2
                    return np.sum(log_pr) + np.sum(log_pr_cosmo)
            else:
                return -np.inf
        else:
            return -np.inf

elif igm_history == "Tfree" and DATASET == "mike-hires":
    
    def lnprior(theta):    
        g = z_dep_func(theta[-2],theta[-1],redshift_list[iflux:])
        if 1. <= g[0] <= 1.7 and 1. <= g[1] <= 1.7 and 1. <= g[2] <= 1.7 and 1. <= g[3] <= 1.7:
            if 6. <= theta[0] <= other_param_max[0] and 0.5 <= theta[1] <= 1.5 and ll_neff <= theta[2] <= ul_neff and 0. <= theta[3] <= 1. and abs(theta[8]-theta[9]) <= 0.5 and abs(theta[9]-theta[10]) <= 0.5 and abs(theta[10]-theta[11]) <= 0.5:
            
                log_pr[:] = -0.5*(theta[4:8] - mu[:])**2/sigma**2
                if cosmo_prior == "flat":
                    return np.sum(log_pr) 
                elif cosmo_prior == "ede" or cosmo_prior == "lcdm" or cosmo_prior == "lya":
                    for i in range(len(mu_cosmo)):
                        log_pr_cosmo[i] = -0.5*(theta[i] - mu_cosmo[i])**2/sigma_cosmo[i]**2
                    return np.sum(log_pr) + np.sum(log_pr_cosmo)
            else:
                return -np.inf
        else:
            return -np.inf
            

elif igm_history == "Tfree" and DATASET != "mike-hires":
    
    def lnprior(theta): 
        g = z_dep_func(theta[-2],theta[-1],redshift_list[iflux:])
        if 1. <= g[0] <= 1.7 and 1. <= g[1] <= 1.7 and 1. <= g[2] <= 1.7 and 1. <= g[3] <= 1.7:
            if 6. <= theta[0] <= other_param_max[0] and 0.5 <= theta[1] <= 1.5 and ll_neff <= theta[2] <= ul_neff and 0. <= theta[3] <= 1. and abs(theta[14]-theta[15]) <= 0.5 and abs(theta[15]-theta[16]) <= 0.5 and abs(theta[16]-theta[17]) <= 0.5 and abs(theta[17]-theta[18]) <= 0.5 and abs(theta[18]-theta[19]) <= 0.5 and abs(theta[19]-theta[20]) <= 0.5 and abs(theta[20]-theta[21]) <= 0.5 and abs(theta[21]-theta[22]) <= 0.5 and abs(theta[22]-theta[23]) <= 0.5: 
            
                log_pr[:] = -0.5*(theta[4:14] - mu[:])**2/sigma**2
                if cosmo_prior == "flat":
                    return np.sum(log_pr) 
                elif cosmo_prior == "ede" or cosmo_prior == "lcdm" or cosmo_prior == "lya":
                    for i in range(len(mu_cosmo)):
                        log_pr_cosmo[i] = -0.5*(theta[i] - mu_cosmo[i])**2/sigma_cosmo[i]**2
                    return np.sum(log_pr) + np.sum(log_pr_cosmo)
            else:
                return -np.inf
        else:
            return -np.inf

    
#### full likelihood (=> likelihood + priors)
def lnprob(theta, z, DATASET):
    ### negative f_uv values very close to zero are fine!
    if abs(theta[3]) <= 1e-3:
        theta[3] = 0.
    
    lp = lnprior(theta); ll = lnlike(theta, z, DATASET)
    if not np.isfinite(lp) or np.isnan(ll):
        return -np.inf
    else:
        return (lp + ll)


nwalkers = 100
start = np.zeros((ndim)); step = np.zeros((ndim))
step[:] = 0.01
start[:4] = other_param_ref[:] #z_reio, s8, neff, fUV
start[-2] = 1.30 #slopeA
start[-1] = 0.25 #slopeS

if igm_history == "Tpowerlaw":
    start[4:n] = flux_ref_old[iflux:] # mean flux F(z)
    start[-4] = 0.8 #T0A
    start[-3] = -3.46 #T0S
    step[-1] = 1.
    step[-3] = 1.

elif igm_history == "Tfree" and DATASET == "mike-hires":
    start[4:8] = flux_ref_old[iflux:] # mean flux F(z)
    start[8:12] = t0_ref_old[6:]*1e-4 #Tz
    
elif igm_history == "Tfree" and DATASET != "mike-hires":
    start[4:14] = flux_ref_old[iflux:] # mean flux F(z)
    start[14:24] = t0_ref_old[:]*1e-4 #Tz

print start

num_burn = 1000
num_steps = 10000 + num_burn

pos = [ np.sum([start,step*np.random.randn(ndim)],axis=0) for i in range(nwalkers) ]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(redshift_list, DATASET), threads=4)
sampler.run_mcmc(pos, num_steps)
#sampler.run_mcmc(pos, 10)

###burn-in and flattening the chain
samples = sampler.chain[:, num_burn:, :].reshape((-1, ndim))

###saving all chisquared
samples2 = sampler.lnprobability[:, num_burn:].reshape((-1))

###print outputs
folder = "./chains/"
if not os.path.exists(folder):
  os.makedirs(folder)

fn = folder+NOMEFILE+"_"+cosmo_prior+"prior_"+DATASET+"_reduced.dat"

final_samples = np.zeros((ndim+1,len(samples2)))

print np.shape(samples), np.shape(samples2), np.shape(final_samples)

final_samples[0,:] = np.around((-1.)*samples2[:],2)
for i in range(ndim):
    final_samples[i+1,:] = np.around(samples[:,i],4)

np.savetxt(fn, np.transpose(final_samples), fmt='%.4f')
