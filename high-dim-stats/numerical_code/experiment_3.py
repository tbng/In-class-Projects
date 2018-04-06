from general_module import *

def experiment_3a(n, p, k=0.018, noise=1.0):
    '''
    X are independent draws from N (0, 1/n).
    k = 0.018 # true signal ratio
    Set beta_1 = ... = beta_18 = 2.5 * np.sqrt(2 * log(p)), beta_19 = ... = beta_1000 = 0
    The errors are independent Gaussian (noisy case) or equal 0 (noiseless)
    '''
    
    X = np.random.normal(size=[n,p], scale = np.sqrt(1/n))
    beta_true = np.concatenate([np.zeros(int(p * k)) + 2.5 * np.sqrt(2 * np.log(p)), np.zeros(int(p * (1 - k)))])
    z = np.random.normal(size=n, scale=noise) # z folows N(0,1) in the noisy case
    y = X.dot(beta_true) + z

    # returns alphas (numpy array of regularization papamters) and coef_path (corresponding coefficients with each alpha)
    alphas, coef_path, _ = lasso_path(X, y, n_alphas=p, eps=0.001) 

    tpp_est = []
    fdp_est = []

    for i in range(coef_path.shape[1]):
        fdp_vs_tpp = fdp_tpp_cal(coef_path[:, i], beta_true)
        tpp_est.append(fdp_vs_tpp[0])
        fdp_est.append(fdp_vs_tpp[1])
        # stop when TPP hits 1.0
        if fdp_vs_tpp[0] == 1.0:
            break
            
    return np.array(tpp_est), np.array(fdp_est)

def experiment_3b(n, p, k=0.018, n_simu=100, noise=True):
    
    percentile = np.arange(0, 102, 2) # percentile used for calculate average of rate
    
    tpp_avg = np.zeros(len(percentile))
    fdp_avg = np.zeros(len(percentile))
    
    # check if noisy or noiseless case
    if noise:
        z = 1.0
    else:
        z = 0.0
        
    for i in range(n_simu):
        tpp_est, fdp_est = experiment_3a(n, p, noise=z)
        tpp_avg += np.percentile(tpp_est, percentile) / n_simu
        fdp_avg += np.percentile(fdp_est, percentile) / n_simu

    return tpp_avg, fdp_avg


# WARNING: the code below will take long time to run due to 200 loops it makes
tpp_6, fdp_6 = asym_tpp_fdp(delta=250/1000, epsi=18/1000)
tpp_noisy_6, fdp_noisy_6 = experiment_3b(250, 1000, n_simu=200, noise=True)
tpp_noiseless_6, fdp_noiseless_6 = experiment_3b(250, 1000, n_simu=200, noise=False)
tpp_est_6, fdp_est_6 = experiment_3a(250, 1000)

g, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(14,6))

ax1.plot(tpp_est_6, fdp_est_6, 'ro')
ax1.plot(tpp_6, fdp_6, 'c', linewidth=2.0)

ax1.legend(['Lasso with $n = 250, p = 1000, \sigma^2_z = 1.0$', 'boundary'])
ax1.set_xlabel('TPP', fontsize=18)
ax1.set_ylabel('FDP', fontsize=18)
ax1.set_xlim([-0.03, 1.03])

ax2.plot(tpp_6, fdp_6, 'c', linewidth=3.0)
ax2.plot(tpp_noisy_6, fdp_noisy_6, '-.', linewidth=3.0)
ax2.plot(tpp_noiseless_6, fdp_noiseless_6, '-.', linewidth=3.0)
ax2.set_xlabel('TPP', fontsize=18)
ax2.set_ylabel('Mean FDP', fontsize=18)
ax2.legend(['boundary', 'mean FDP, noisy case', 'mean FDP, noiseless case'])

g.tight_layout()
# g.savefig('figure6-paper.pdf') # uncomment to save figure

plt.show()
plt.close('all')
