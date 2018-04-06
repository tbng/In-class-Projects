from general_module import *

# function that do experiment 1
def experiment_1(size_X):
    '''
    size_X is a list of [n, p]
    X are independent draws from N (0, 1).  
    Set beta_1 = ... = beta_200 = 4, beta_201 = ... = beta_1000 = 0
    The errors are independent Gaussian.
    '''
    n, p = size_X
    X = np.random.normal(size=[n,p])
    k = 0.2 # true signal ratio
    beta_true = np.concatenate([np.zeros(int(p * k)) + 4, np.zeros(int(p * (1 - k)))])
    error = np.random.normal(size=n) # error folows N(0,1)
    y = X.dot(beta_true) + error

    # returns alphas (numpy array of regularization papamters) and coef_path
    # (corresponding coefficients with each alpha)
    alphas, coef_path, _ = lasso_path(X, y, n_alphas=p, eps=0.001, n_jobs=3) 

    tpp_est = []
    fdp_est = []

    for i in range(coef_path.shape[1]):
        fdp_vs_tpp = fdp_tpp_cal(coef_path[:, i], beta_true)
        tpp_est.append(fdp_vs_tpp[0])
        fdp_est.append(fdp_vs_tpp[1])
        # stop when TPP hits 1.0
        if fdp_vs_tpp[0] == 1.0:
            break
            
    tpp_est = np.array(tpp_est)
    fdp_est = np.array(fdp_est)

    return tpp_est, fdp_est

tpp, fdp = asym_tpp_fdp(delta=1010/1000, epsi=0.2)
tpp_est, fdp_est = experiment_1(size_X=[1010, 1000])

plt.figure(figsize=(10,6))
plt.plot(tpp_est, fdp_est, 'o', markersize=5.0)
plt.plot(tpp, fdp, 'c', linewidth=1.5)
plt.legend(['Lasso', 'Asymptotic trade-off curve'])
plt.xlabel('TPP', fontsize=18)
plt.ylabel('FDP', fontsize=18)
# plt.savefig('figure1-paper.pdf') # uncomment to save figure
plt.show()

'''
# WARNING: this part below taks quite a long time to run
tpp_est_mc = []
fdp_est_mc = []

for i in range(2):
    tpp_simu, fdp_simu = experiment_1(size_X=[1010, 1000])
    tpp_est_mc.append(tpp_simu[fdp_simu > 0][0]) # TPP at time of first false selection
    fdp_est_mc.append(fdp_simu[-1]) # FDP at time of last true selection

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

ax1.hist(tpp_est_mc, bins=list(np.arange(0.0, 0.4, 0.03)), ec='white')
ax1.set_xlabel('TPP at time of first false selection', fontsize=18)
ax1.set_ylabel('Frequency', fontsize=18)

ax2.hist(fdp_est_mc, bins=list(np.arange(0.0, 0.30, 0.02)), ec='white')
ax2.set_xlabel('FDP at time of last true selection', fontsize=18)

# plt.savefig('figure2-paper.pdf') # uncomment to save figure
plt.show()
'''
plt.close('all')

