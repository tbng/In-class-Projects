from general_module import *

def experiment_2a(n, p, delta=1.0, epsi=0.2):
    
    coef = np.random.choice([0.0, 50], size=p, p=[1 - epsi, epsi]) # generate coef follows $\mathbb{P}(\Pi = 50) = 1 - \mathbb{P}(\Pi = 0) = \epsilon$
    X = np.random.normal(size=[n,p], scale=np.sqrt(1/n)) # design matrix X has iid N(0, 1/n)
    z = 0 # noiseless case
    y = X.dot(coef) + z
    
    # returns alphas (numpy array of regularization papamters) and coef_path (corresponding coefficients with each alpha)
    alphas, coef_path, _ = lasso_path(X, y, n_alphas=p, eps=0.001) # eps is different from epsilon in paramter
    tpp_est = []
    fdp_est = []
    
    for i in range(coef_path.shape[1]):
        fdp_vs_tpp = fdp_tpp_cal(coef_path[:, i], coef)
        tpp_est.append(fdp_vs_tpp[0])
        fdp_est.append(fdp_vs_tpp[1])
        # stops when TPP hits 1.0
        if fdp_vs_tpp[0] == 1.0:
            break
            
    tpp_est = np.array(tpp_est)
    fdp_est = np.array(fdp_est)
    return tpp_est, fdp_est

def experiment_2b(n, p, epsi_prime, delta=1.0, epsi=0.2):
    coef = np.random.choice([0.0, 0.1, 50], size=p, 
                            p=[1 - epsi, epsi * (1 - epsi_prime), epsi_prime * epsi]) # generate coef follows 3 points prior
    X = np.random.normal(size=[n,p], scale=np.sqrt(1/n)) # design matrix X has iid N(0, 1/n)
    z = 0 # noiseless case
    y = X.dot(coef) + z
    
    # returns alphas (numpy array of regularization papamters) and coef_path (corresponding coefficients with each alpha)
    alphas, coef_path, _ = lasso_path(X, y, n_alphas=p, eps=0.0001, n_jobs=-1) # eps is different from epsilon in paramter
    tpp_est = []
    fdp_est = []
    
    for i in range(coef_path.shape[1]):
        fdp_vs_tpp = fdp_tpp_cal(coef_path[:, i], coef)
        tpp_est.append(fdp_vs_tpp[0])
        fdp_est.append(fdp_vs_tpp[1])
        # stops when TPP hits 1.0
        if fdp_vs_tpp[0] == 1.0:
            break
            
    tpp_est = np.array(tpp_est)
    fdp_est = np.array(fdp_est)
    return tpp_est, fdp_est

def experiment_2b_mc(n, p, epsi_prime, delta=1.0, epsi=0.2, n_simu=50):

    percentile = np.arange(0, 102, 2) # percentile used for calculate average of rate

    tpp_avg = np.zeros(len(percentile))
    fdp_avg = np.zeros(len(percentile))

    for i in range(n_simu):
        tpp_est, fdp_est = experiment_2b(n, p, epsi_prime)
        tpp_avg += np.percentile(tpp_est, percentile) / n_simu
        fdp_avg += np.percentile(fdp_est, percentile) / n_simu

    return tpp_avg, fdp_avg


# WARNING: this experiment will take very long time to run

tpp_5, fdp_5 = asym_tpp_fdp(delta=1.0, epsi=0.2)

f, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(16,6))

ax1.plot(tpp_5, fdp_5, 'c', linewidth=1.5)
ax1.legend(['boundary'])

for i in range(10):
    tpp_est_1000, fdp_est_1000 = experiment_2a(1000, 1000)
    ax1.plot(tpp_est_1000, fdp_est_1000, 'ro', markersize=4.0)

ax1.set_xlabel('TPP', fontsize=18)
ax1.set_ylabel('FDP', fontsize=18)
ax1.set_xlim([-0.03, 1])
ax1.set_ylim([-0.01, 0.25])

list_epsi_prime = [0.3, 0.5, 0.7, 0.9]
color_list = ['red', 'blue', 'black', 'magenta']

# WARNING: this loop will take a long time to run
for i in range(len(list_epsi_prime)):
    tpp_avg, fdp_avg = experiment_2b_mc(1000, 1000, list_epsi_prime[i], n_simu=100)
    plt.plot(tpp_avg, fdp_avg, linestyle='--', color=color_list[i], linewidth=1.5)

ax2.plot(tpp_5, fdp_5, 'c', linewidth=3.0)

ax2.set_xlabel('Mean TPP', fontsize=18)
ax2.set_ylabel('Mean FDP', fontsize=18)
ax2.legend(['$\epsilon\'$ = 0.3', '$\epsilon\'$ = 0.5', '$\epsilon\'$ = 0.7',
            '$\epsilon\'$ = 0.9', 'boundary'])

f.savefig('figure5-paper.pdf') # uncomment to save figure
plt.show()
plt.close('all')
