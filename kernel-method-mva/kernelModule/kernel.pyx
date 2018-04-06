import numpy as np
cimport numpy as np
import itertools # for making combination of sequence dictionary with spectrum kernel

from numpy cimport ndarray
from scipy.linalg import norm
from libc.math cimport sqrt, exp
from skbio.alignment import local_pairwise_align_ssw
from skbio import DNA

cdef class KernelPairwise(object):
    """ Transform a list of DNA sequences used for training to pairwise kernel matrix
    and transform test DNA sequence to equivalent test pairwise kernel matix following
    Liao and Noble (2003)
    """
    cdef public:
        int n_train, match_score, mismatch_score, gap_open_penalty, gap_extend_penalty
        double gamma
        ndarray train_data, train_mean, train_std
        double[:, :] pw_matrix
        str method
        shift_norm
        use_linear_normalized

    def __cinit__(self, gamma=1e-3, match_score=1, mismatch_score=-1, gap_open_penalty=11,
                  gap_extend_penalty=1, method='normalized', shift_norm=True,
                  use_linear_normalized=True):
        """ train_data must be a numpy array list of DNA sequence
        ----------
        Input
        ----------
            ref_dict: dictionary of all possible permutation of k-mer of DNA sequences
            train_data: numpy array of string of DNA sequences used for training the model
            n_train: number of train observations
            shift_norm: if True then the value of pairwise score matrix will be normalized to 0 mean and 1 std first

        """
        self.match_score, self.mismatch_score, = match_score, mismatch_score
        self.gap_open_penalty, self.gap_extend_penalty =  gap_open_penalty, gap_extend_penalty
        self.method, self.shift_norm = method, shift_norm
        self.use_linear_normalized = use_linear_normalized
        self.gamma = gamma

    cpdef cal_pairwise_score(self, x, y):
        """Compute the pairwise vector score of observations matrix using Smith_Waterman pairwise function from skbio
        In other words we transform a vector of string DNA sequence to a numeric vector 

        Input:
            x and y: numpy array of DNA string sequence
        Output:
            return the SW similarity score between x and y
        """
        cdef double similarity_score
        similarity_score = local_pairwise_align_ssw(DNA(x), DNA(y),
                                                    match_score=self.match_score,
                                                    mismatch_score=self.mismatch_score,
                                                    gap_open_penalty=self.gap_open_penalty,
                                                    gap_extend_penalty=self.gap_extend_penalty)[1]

        return similarity_score

    cpdef cal_score_normalized(self, double[:] x, double[:] y):
        """Normalized a pair of vectors of similarity score  by the formular K(X, Y) = X.Y / sqrt( X.X * Y.Y )
        Follow method in Liao & Noble (2003)
        ----------
        Input
        ----------
            x and y: numpy.array
        ----------
        Output
        ----------
            normalized score (float)
        """
        cdef double normalized_score
        normalized_score = np.dot(x, y) / sqrt(np.dot(x, x) * np.dot(y, y))

        return normalized_score

    cpdef cal_score_rbf(self, double[:] x, double[:] y, double gamma):
        """ Calculate the radial basis function kernel for vector x and y
        Formula: K(x, y) = exp(-gamma * norm(x - y) ** 2)
        
        Input
        ----------
            x and y: numpy array store as cython memoryview
            gamma: hyperparameters for the RBF kernel
        Output
        ----------
            RBF score
        """
        cdef double rbf_score

        rbf_score = exp(-gamma * norm(np.asarray(x) - np.asarray(y)) ** 2)

        return rbf_score

    cpdef pairwise_kernel_matrix(self, ndarray list_seq_1, ndarray list_seq_2, shift_norm=False):
        """Compute the kernel matrix of observations matrix using pairwise_vector_score
        ----------
        Input
        ----------
            list_seq_1, list_seq_2: numpy array
            shift_norm: boolean, whether to shift the data to zero mean and unit std or not
        ----------
        Output
        ----------
            return the numpy matrix of pairwise score between sequence a and b in data
        """
        cdef double[:, :] pw_matrix
        cdef int i, j, nseq1, nseq2
        nseq1 = len(list_seq_1)
        nseq2 = len(list_seq_2)
        pw_matrix = np.zeros(shape=(nseq1, nseq2))

        for i in range(nseq1):
            for j in range(nseq2):
                pw_matrix[i, j] = self.cal_pairwise_score(list_seq_1[i], list_seq_2[j])

        return pw_matrix

    cpdef fit_train_kernel(self, ndarray train_data):
        """Normalized a kernel matrix by the formular K(X, Y) = X.Y / sqrt( X.X * Y.Y )
        """
        cdef double[:, :] train_matrix
        cdef int i, j

        # update value of train_data and n_train
        self.train_data = train_data
        self.n_train = len(train_data)
        self.pw_matrix = self.pairwise_kernel_matrix(self.train_data, self.train_data)
        self.train_mean = np.mean(self.pw_matrix, axis=0)
        self.train_std = np.std(self.pw_matrix, axis=0)
        # shift raw pw_matrix to have zero mean and unit variance
        if self.shift_norm == True:
            self.pw_matrix = (self.pw_matrix - self.train_mean) / (self.train_std)

        train_matrix = np.zeros(shape=(self.n_train, self.n_train))

        if self.method == 'normalized':
            for i in range(self.n_train):
                for j in range(self.n_train):
                    if self.use_linear_normalized == True:
                        train_matrix[i, j] = self.cal_score_normalized(self.pw_matrix[i, :], self.pw_matrix[j, :])
                    elif self.use_linear_normalized == False:
                        train_matrix[i, j] = self.pw_matrix[i, j] / sqrt(self.pw_matrix[i, i] * self.pw_matrix[j, j])
                    elif self.use_linear_normalized == None:
                        train_matrix = self.pw_matrix.copy()

        elif self.method == 'rbf':
            for i in range(self.n_train):
                for j in range(self.n_train):
                    train_matrix[i, j] = self.cal_score_rbf(self.pw_matrix[i, :], self.pw_matrix[j, :], gamma=self.gamma)
        else:
            print('Error: invalid chosen method.')

        return train_matrix

    cpdef fit_test_kernel(self, ndarray test_seq):
        """Calculate normalized pairwise ready for prediction of svm
        ----------
        Input
        ----------
            test_seq: sequence want to make pairwise score (e.g. xte1)
        ----------
        """
        cdef int n_test = len(test_seq), i, j
        cdef double[:, :] pw_score_temp, test_matrix

        pw_score_temp = self.pairwise_kernel_matrix(test_seq, self.train_data,
                                                    shift_norm=self.shift_norm)
        # shift pw_score_temp using same scale as transforming train data
        if self.shift_norm == True:
            pw_score_temp = (pw_score_temp - self.train_mean) / (self.train_std)

        test_matrix = np.zeros(shape=(n_test, self.n_train))

        if self.method == 'normalized':
            for i in range(n_test):
                for j in range(self.n_train):
                    if self.use_linear_normalized == True:
                        test_matrix[i, j] = self.cal_score_normalized(pw_score_temp[i, :], self.pw_matrix[j, :])
                    elif self.use_linear_normalized == False:
                        test_matrix[i, j] = pw_score_temp[i, j] / sqrt(pw_score_temp[i, i] * self.pw_matrix[j, j])
                    elif self.use_linear_normalized == None:
                        test_matrix = pw_score_temp.copy()

        elif self.method == 'rbf':
            for i in range(n_test):
                for j in range(self.n_train):
                    test_matrix[i, j] = self.cal_score_rbf(pw_score_temp[i, :], self.pw_matrix[j, :], gamma=self.gamma)
        else:
            print('Error: invalid chosen method.')

        return test_matrix

#############################################################################################################################

cdef class KernelSpectrum(object):
    """Transform a list of DNA sequences used for training to spectrum kernel matrix of k-grams
    and transform test DNA sequence to test spectrum kernel matix follow Leslie (2001)
    """
    cdef public:
        int k, n_train
        ref_dict, method
        ndarray raw_train
        double gamma

    def __cinit__(self, k=4, gamma=1e-3, method='normalized'):
        """ train_data must be a numpy array list of DNA sequence
        ----------
        Input
        ----------
            ref_dict: dictionary of all possible permutation of k-mer of DNA sequences
            train_data: numpy array of string of DNA sequences used for training the model
            n_train: number of train observations

        """
        self.k = k
        # building a reference suffix tree
        ref_dict = list(itertools.product(['A', 'T', 'C', 'G'], repeat=self.k)) # this is a list of tuple
        self.ref_dict = [''.join(tup) for tup in ref_dict] # join a list of tuple to become string of k-gram sequence
        self.method = method
        self.gamma = gamma

    cpdef cal_spectrum_score(self, seq):
        """Return a list of size 4^k score for each sequence, with each element being the number of appearance of k-gram
        ----------
        Input
        ----------
            seq: list of string of DNA sequence
            
        ----------
        Output
        ----------
            4^k length numpy array of score
        """
        score_vect = [seq.count(word) for word in self.ref_dict]

        return np.array(score_vect)

    cpdef ndarray raw_spec_matrix(self, ndarray list_seq):
        cdef int n_seq = len(list_seq)
        cdef raw_score
        raw_score = []

        for i in range(n_seq):
            raw_score.append(self.cal_spectrum_score(list_seq[i]))

        return np.array(raw_score)

    cpdef cal_kern_score(self, ndarray spec_vect_1, ndarray spec_vect_2):
        """
        ----------
        Input
        ----------
            seq1, seq2: string of DNA sequences
        ----------
        Output
        ----------
            Spectrum kernel score of 2 sequences 
        """

        if self.method == 'normalized':
            return np.dot(spec_vect_1, spec_vect_2) / sqrt(np.dot(spec_vect_1, spec_vect_1) * np.dot(spec_vect_2, spec_vect_2))
        elif self.method == 'rbf':
            return exp(- self.gamma * norm(spec_vect_1 - spec_vect_2) ** 2)
        elif self.method == 'raw':
            return np.dot(spec_vect_1, spec_vect_2)

    cpdef fit_train_kernel(self, ndarray train_data):
        """ Fit the self.train_data into spectrum kernel matrix 
        ----------
        Input
        ----------
            
        ----------
        Output
        ----------
            numpy matrix of size n_train x n_train
        """
        cdef double[:, :] train_matrix
        cdef int i, j

        # self.train_data = train_data
        self.n_train = len(train_data)
        self.raw_train = self.raw_spec_matrix(train_data)

        train_matrix = np.zeros(shape=(self.n_train, self.n_train))

        for i in range(self.n_train):
            for j in range(self.n_train):
                train_matrix[i, j] = self.cal_kern_score(self.raw_train[i, :], self.raw_train[j, :])

        return train_matrix

    cpdef fit_test_kernel(self, test_seq):
        """
        ----------
        Input
        ----------
            test_seq: numpy array of string DNA sequences
            self
        ----------
        Output
        ----------
            numpy matrix of size n_test x n_train
        """
        cdef double[:, :] test_matrix
        cdef ndarray raw_test
        cdef int i, j

        n_test = len(test_seq)
        test_matrix = np.zeros(shape=(n_test, self.n_train))
        raw_test = self.raw_spec_matrix(test_seq)

        for i in range(n_test):
            for j in range(self.n_train):
                test_matrix[i, j] = self.cal_kern_score(raw_test[i, :], self.raw_train[j, :])

        return test_matrix


#############################################################################################################################


cdef class KernelWDwithShift(object):
     """ Transform a list of DNA sequences used for training to WD with shift kernel
    matrix and transform test DNA sequence to equivalent WD kernel matix
    following Sonnenburg et al (2005)
    """
    cdef public:
        int d, S, n_train
        ndarray train_data, wd_matrix

    def __cinit__(self, int d=7, int S=1):
        self.n_train, self.d, self.S = 0, d, S

    cpdef gen_kmer_list(self, str seq, int k):
        """Generate a list of k-spectrum
        seq: a string of DNA sequences
        """
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

    cpdef cal_wd_score(self, str seq1, str seq2):
        """ weighted degree score as in 4.2.3 of Large Scale Machine learning
        parameters
        ----------
        d : maximum length of k-mers
        S : maximum gap allowed
        """
        cdef double score = 0, score_k, beta_k, mu_s, delta_s
        cdef int d = self.d
        cdef int S = self.S
        cdef int mer_len, mer_s, k, s, i

        for k in range(1, d + 1):
            score_k = 0
            beta_k = 2 * (d - k + 1) / (d * (d + 1))

            mer_list_1 = self.gen_kmer_list(seq1, k=k)
            mer_list_2 = self.gen_kmer_list(seq2, k=k)

            mer_len = len(mer_list_1)

            for s in range(S + 1):
                mu_s = 0
                delta_s = 1 / (2 * (s + 1))
                mer_s = mer_len - s

                for i in range(mer_s):
                    mu_s += float(mer_list_1[i + s] == mer_list_2[i]) + float(mer_list_1[i] == mer_list_2[i + s])

                score_k += delta_s * mu_s

            score += beta_k * score_k

        return score

    cpdef wd_score_mat(self, ndarray seq_list_1, ndarray seq_list_2):

        cdef ndarray wd_matrix
        cdef int n1, n2, i, j
        n1 = len(seq_list_1)
        n2 = len(seq_list_2)
        wd_matrix = np.zeros(shape=(n1, n2))

        for i in range(n1):
            for j in range(n2):
                wd_matrix[i, j] = self.cal_wd_score(str(seq_list_1[i]), str(seq_list_2[j]))

        return wd_matrix

    cpdef fit_train_kernel(self, ndarray train_data):
        self.train_data = train_data
        self.n_train = len(train_data)
        cdef int i, j
        cdef double[:, :] train_matrix = np.zeros(shape=(self.n_train, self.n_train))

        self.wd_matrix = self.wd_score_mat(train_data, train_data)

        for i in range(self.n_train):
            for j in range(self.n_train):
                train_matrix[i, j] = self.wd_matrix[i, j] / sqrt(self.wd_matrix[i, i] * self.wd_matrix[j, j])

        return train_matrix

    cpdef fit_test_kernel(self, ndarray test_seq):
        """
        ----------
        Input
        ----------
            test_seq: numpy array of string DNA sequences
            self
        ----------
        Output
        ----------
            numpy matrix of size n_test x n_train
        """
        cdef int n_test = len(test_seq), i, j
        cdef ndarray wd_test
        cdef double[:, :] test_matrix = np.zeros(shape=(n_test, self.n_train))
        cdef seq_i, seq_j

        wd_test = self.wd_score_mat(test_seq, self.train_data)

        for i in range(n_test):
            for j in range(self.n_train):
                test_matrix[i, j] = wd_test[i, j] / sqrt(wd_test[i, i] * self.wd_matrix[j, j])

        return test_matrix
