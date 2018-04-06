"""
This code will produce our final submission result by ensembling 5 models described in our report
"""
import numpy as np
import pandas as pd
from scipy import stats
#import sys
#sys.path.append('./')

from kernelModule.kernel import KernelPairwise, KernelSpectrum, KernelWDwithShift
from kernelModule.svm import SVMclassifier, grid_search_c
from kernelModule.additional_module import data_process, timer, save_csv

if __name__ == "__main__":

    with timer('Import data'):
        xtrain, ytrain, xtest, xcv_sand, ycv_sand = data_process(path='./data/')
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

    # Linear normalized spectrum kernel
    with timer('Training linear spectrum kernel'):
        spec_linear = KernelSpectrum(k=4, method='normalized')
        train_matrix = np.asarray(spec_linear.fit_train_kernel(xtrain))
        cv_matrix = np.asarray(spec_linear.fit_test_kernel(xcv_sand))
        test_matrix = np.asarray(spec_linear.fit_test_kernel(xtest))
        c_list = np.arange(1.0, 15, 0.2)
        c_opt, accu_list = grid_search_c(train_matrix, cv_matrix,
                                         label=ytrain, cv_label=ycv_sand,
                                         c_list=c_list)
        svm_offi = SVMclassifier(C=c_opt)
        svm_offi.fit(train_matrix, ytrain)
        pred = svm_offi.predict(test_matrix)
        predictions_3 = pd.DataFrame({'id': np.arange(n_test), 'Bound': pred})
        predictions_3 = predictions_3[['id', 'Bound']]
        predictions_3[predictions_3 == -1.0] = 0.0

    # Spectrum kernel with Gaussian kernel transform
    with timer('Training RBF spectrum kernel'):
        spec_rbf = KernelSpectrum(k=4, method='rbf', gamma=1e-3)
        train_matrix = np.asarray(spec_rbf.fit_train_kernel(xtrain))
        cv_matrix = np.asarray(spec_rbf.fit_test_kernel(xcv_sand))
        test_matrix = np.asarray(spec_rbf.fit_test_kernel(xtest))
        c_list = np.logspace(-3, 4, 7)
        c_opt, accu_list = grid_search_c(train_matrix, cv_matrix,
                                         label=ytrain, cv_label=ycv_sand,
                                         c_list=c_list)
        svm_offi = SVMclassifier(C=c_opt)
        svm_offi.fit(train_matrix, ytrain)
        pred = svm_offi.predict(test_matrix)
        predictions_4 = pd.DataFrame({'id': np.arange(n_test), 'Bound': pred})
        predictions_4 = predictions_4[['id', 'Bound']]
        predictions_4[predictions_4 == -1.0] = 0.0

    # WD kernel with shift normalized
    with timer('Training WD kernel with shift'):
        wd = KernelWDwithShift(d=6, S=3)
        train_matrix = np.asarray(wd.fit_train_kernel(xtrain))
        cv_matrix = np.asarray(wd.fit_test_kernel(xcv_sand))
        test_matrix = np.asarray(wd.fit_test_kernel(xtest))
        c_list = np.arange(1.0, 15, 0.2)
        svm_offi = SVMclassifier(C=c_opt)
        svm_offi.fit(train_matrix, ytrain)
        pred = svm_offi.predict(test_matrix)
        predictions_5 = pd.DataFrame({'id': np.arange(n_test), 'Bound': pred})
        predictions_5 = predictions_5[['id', 'Bound']]
        predictions_5[predictions_5 == -1.0] = 0.0

    # ensembling by majority voting
    with timer('Doing majority voting'):
        ensem = pd.DataFrame(np.arange(3000), columns=['id'])
        ensem['pred_pw'] = predictions_1['Bound']
        ensem['pred_pw_rbf'] = predictions_2['Bound']
        ensem['pred_spec'] = predictions_3['Bound']
        ensem['pred_spec_rbf'] = predictions_4['Bound']
        ensem['pred_wd'] = predictions_5['Bound']
        predictions_final = pd.DataFrame(np.arange(3000), columns=['id'])
        predictions_final['Bound'] = pd.DataFrame(stats.mode(ensem.iloc[:, 1:], axis=1)[0])
        save_csv(predictions_final, 'predictions_final', fl_format='%d')

    print('success !')

