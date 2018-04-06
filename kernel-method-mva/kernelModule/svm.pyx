import numpy as np
cimport numpy as np
import cvxopt # for quadratic solver
from numpy cimport ndarray

cdef class SVMclassifier(object):

    cdef public:
        double C, bias
        double[:] result_alpha
        ndarray label

    def __cinit__(self, double C=1.0):
        self.C = C

    cpdef fit(self, ndarray kernel_matrix, ndarray y):
        """Given the kernel matrix X with labels y, returns a SVM predictor representing the trained SVM.
        Setup follow http://cvxopt.org/userguide/coneprog.html and the book Kernel Methods in Computational biology.
        Notice the problem stated in the slide is maximize, but the qp solver only works with minimization problem.

        The code is highly inspired from http://tullo.ch/articles/svm-py/
        -----------
        Input
        -----------
            kernel_matrix: kernel matrix of size (n_train, n_train), precalculated from the train data
            y: numpy.array, vector of training labels
            C: slack variables for svm, equivalent with regularization parameter
        -----------
        Output
        -----------
            numpy array of alphas
        """
        cdef int n
        cdef ndarray result_alpha
        cdef double[:] raw_result

        self.label = y
        n = len(y) # number of observations

        P = cvxopt.matrix(np.outer(y, y) * kernel_matrix)
        q = cvxopt.matrix(-1.0 * np.ones(n))

        # inequality constraint
        G_upper = cvxopt.matrix(np.diag(np.ones(n)))
        h_upper = cvxopt.matrix(self.C * np.ones(n)) # upper bound is C

        G_lower = cvxopt.matrix(np.diag(-1 * np.ones(n)))
        h_lower = cvxopt.matrix(np.zeros(n)) # lower bound is 0

        G = cvxopt.matrix(np.vstack((G_upper, G_lower)))
        h = cvxopt.matrix(np.vstack((h_upper, h_lower)))

        # equality constraint
        A = cvxopt.matrix(y, (1, n))
        b = cvxopt.matrix(0.0)

        # use cvxopt to solve problem
        cvxopt.solvers.options['show_progress'] = False # silent the output
        sols = cvxopt.solvers.qp(P, q, G, h, A, b)

        result_alpha = np.ravel(sols['x'])
        # reduce the very small weights to 0
        result_alpha[result_alpha < 1e-6] = 0.0
        self.result_alpha = result_alpha

        # calculate the bias term
        raw_result = np.dot(kernel_matrix, self.result_alpha * y)
        self.bias = np.mean(y - raw_result)

    cpdef predict(self, ndarray test_kernel_matrix):

        cdef double[:] result

        result = self.bias + np.dot(test_kernel_matrix, self.result_alpha * self.label)

        return np.sign(result)

    cpdef score(self, ndarray test_mat, ndarray true_label_vector):
        """ Accuracy score of predicted vs. true label
        Input:
        ---------
            predict_vector: numpy array of label of value {-1 ,1}
            true_label_vector: numpy array of label of value {-1, 1}
        Output:
        ---------
            Accuracy score between 0 and 1
        """
        cdef ndarray correct, predict_vector

        assert test_mat.shape[0] == len(true_label_vector), 'Error: two data does not share the same length.'

        predict_vector = self.predict(test_mat)

        correct = np.multiply(predict_vector, true_label_vector) == 1

        return np.sum(correct) / len(predict_vector)

# Define a funtion to tune parameter C
cpdef grid_search_c(ndarray train_kernel,
                    ndarray cv_kernel,
                    ndarray label, ndarray cv_label,
                    ndarray c_list=np.arange(0.5, 10, 0.5)):
    """For grid search tuning of hyperparamter c
    """
    cdef result_list, accu_list
    cdef double c_opt
    cdef int c_ind

    result_list = []
    accu_list = []

    for c in c_list:
        clf = SVMclassifier(C=c)
        clf.fit(train_kernel, label)
        accu_list.append(clf.score(cv_kernel, cv_label))

    c_ind = np.argmax(accu_list)
    c_opt = c_list[c_ind]

    return c_opt, accu_list