"""
Run training process with a small proportion of train and test data for Pairwise kernel only
"""
import numpy as np
import pandas as pd
from scipy import stats

from kernelModule.kernel import KernelPairwise, KernelSpectrum, KernelWDwithShift
from kernelModule.svm import SVMclassifier, grid_search_c
from kernelModule.additional_module import data_process, timer, save_csv

if __name__ == "__main__":

    with timer('Import data'):
        xtrain, ytrain, xtest, xcv_sand, ycv_sand = data_process(path='./data/')
        xtrain, ytrain, xtest = xtrain[:200], ytrain[:200], xtest[:150]
        xcv_sand, ycv_sand = xcv_sand[:100], ycv_sand[:100]
        n_train = xtrain.shape[0]
        n_test = xtest.shape[0]

    # Linear normalized Pairwise kernel
    with timer('Training linear pairwise kernel'):
        pw_linear = KernelPairwise()
        train_matrix = np.asarray(pw_linear.fit_train_kernel(xtrain))
        cv_matrix = np.asarray(pw_linear.fit_test_kernel(xcv_sand))
        test_matrix = np.asarray(pw_linear.fit_test_kernel(xtest))
        c_list = np.arange(1.0, 15, 0.2)
        c_opt, accu_list = grid_search_c(train_matrix, cv_matrix,
                                         label=ytrain, cv_label=ycv_sand,
                                         c_list=c_list)
        svm_offi = SVMclassifier(C=c_opt)
        svm_offi.fit(train_matrix, ytrain)
        pred = svm_offi.predict(test_matrix)
        predictions_1 = pd.DataFrame({'id': np.arange(n_test), 'Bound': pred})
        predictions_1 = predictions_1[['id', 'Bound']]
        predictions_1[predictions_1 == -1.0] = 0.0

    # Pairwise kernel with Gaussian kernel transform
    with timer('Training RBF pairwise kernel'):
        pw_rbf = KernelPairwise(gamma=1e-3, method='rbf')
        train_matrix = np.asarray(pw_rbf.fit_train_kernel(xtrain))
        cv_matrix = np.asarray(pw_rbf.fit_test_kernel(xcv_sand))
        test_matrix = np.asarray(pw_rbf.fit_test_kernel(xtest))
        c_list = np.logspace(-3, 4, 7)
        c_opt, accu_list = grid_search_c(train_matrix, cv_matrix,
                                         label=ytrain, cv_label=ycv_sand,
                                         c_list=c_list)
        svm_offi = SVMclassifier(C=c_opt)
        svm_offi.fit(train_matrix, ytrain)
        pred = svm_offi.predict(test_matrix)
        predictions_2 = pd.DataFrame({'id': np.arange(n_test), 'Bound': pred})
        predictions_2 = predictions_2[['id', 'Bound']]
        predictions_2[predictions_2 == -1.0] = 0.0


    # ensembling by majority voting
    with timer('Doing majority voting'):
        ensem = pd.DataFrame(np.arange(3000), columns=['id'])
        ensem['pred_pw'] = predictions_1['Bound']
        ensem['pred_pw_rbf'] = predictions_2['Bound']
        predictions_final = pd.DataFrame(np.arange(3000), columns=['id'])
        predictions_final['Bound'] = pd.DataFrame(stats.mode(ensem.iloc[:, 1:], axis=1)[0])
        save_csv(predictions_final, 'predictions_final', fl_format='%d')

    print('success !')

