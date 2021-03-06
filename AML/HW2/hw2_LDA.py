import sys
import timeit
import numpy as np
from scipy.special import gamma
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# For the sampling of alpha and beta, we use a baby version of Metropolis-Hastings Sampler
# By default, set the Gamma distribution as proposal distribution
def abSampler(x0, x_id, count_mat, n_burn, a, b):
    length = len(count_mat)
    for t in range(0, n_burn):
        x_c = np.random.gamma(shape=x0)
        log_u = np.log(np.random.random(size=1))
        log_acc_ratio = np.sum([np.log(gamma(count_mat[i, x_id]+x_c)*gamma(x0)/gamma(x_c)/gamma(count_mat[i, x_id]+x0))
                                +(a-1)*np.log(x_c/x0)-b*(x0-x_c) for i in range(0, length)])
        log_acc_ratio += (x_c-1)*np.log(x0) - (x0-1)*np.log(x_c) - (x0-x_c) + np.log(gamma(x0)/gamma(x_c))
        if log_u <= log_acc_ratio:
            x0 = x_c
    # sample one x
    x_c = np.random.gamma(shape=x0)
    log_u = np.log(np.random.random(size=1))
    log_acc_ratio = np.sum([np.log(gamma(count_mat[i, x_id]+x_c)*gamma(x0)/gamma(x_c)/gamma(count_mat[i, x_id]+x0))
                                +(a-1)*np.log(x_c/x0)-b*(x0-x_c) for i in range(0, length)])
    log_acc_ratio += (x_c - 1) * np.log(x0) - (x0 - 1) * np.log(x_c) - (x0 - x_c) + np.log(gamma(x0) / gamma(x_c))
    if log_u <= log_acc_ratio:
        return x_c
    else:
        return x0


# Latent Dirichlet Allocation model solved by Gibbs sampling
class ldaModel(object):

    def __init__(self, n_topics, a, b):
        self.K = n_topics
        self.a = a
        self.b = b

    def fit(self, data, Vmax, n_sample, n_burn, hyper_update=False):
        # data: word frequency matrix for the corpus
        # where data[m, t] = the amount of t-th words in m-th document
        M = len(data)
        # initialize the hidden variables
        # ---assign topic mixture weight and vocabulary distribution
        Theta = np.zeros((M, self.K))
        Phi = np.zeros((self.K, Vmax))
        # ---assign topics randomly
        # the number of times topic z_k occurs in document d_m
        c_topic = np.zeros((M, self.K))
        # the number of times word w_t is generated by topic z_k
        c_word = np.zeros((self.K, Vmax))
        Z = np.zeros((M, Vmax), dtype=np.int64)
        for m in range(0, M):
            words_m = np.nonzero(data[m])[0]
            for n in words_m:
                tp_new = np.random.randint(low=1, high=self.K+1)
                Z[m, n] = tp_new
                c_topic[m, tp_new-1] += 1
                c_word[tp_new-1, n] += 1

        # initialize the hyper-parameters
        alpha = np.ones(self.K)*50/self.K
        beta = np.ones(Vmax)*0.01
        topic_set = np.array([i for i in range(1, self.K+1)])

        # Fill Z by Gibbs sampling from posterior distribution
        # ---Start the burn-in process of Gibbs sampling
        for t_burn in range(0, n_burn):
            for m in range(0, M):
                words_m = np.nonzero(data[m])[0]
                for w_id in words_m:
                    w_tp = Z[m, w_id]
                    c_topic[m, w_tp-1] -= 1
                    c_word[w_tp-1, w_id] -= 1
                    # ------select a new topic
                    z_pval = np.array([(c_word[k, w_id]+beta[w_id])*(c_topic[m, k]+alpha[k])/(np.sum(c_word[k, :])+np.sum(beta))
                                        for k in range(0, self.K)])
                    z_pval = z_pval/np.sum(z_pval)
                    tp_new = np.random.choice(topic_set, 1, p=z_pval)[0]
                    Z[m, w_id] = tp_new
                    # ------update the counter matrix
                    c_topic[m, tp_new-1] += 1
                    c_word[tp_new-1, w_id] += 1
                    # Update the hyper-parameters
                    if hyper_update:
                        for k in range(0, self.K):
                            alpha[k] = abSampler(x0=alpha[k], x_id=k, count_mat=c_topic, n_burn=1000, a=self.a, b=self.b)
                        for v in range(0, Vmax):
                            beta[v] = abSampler(x0=beta[v], x_id=v, count_mat=c_word, n_burn=1000, a=self.a, b=self.b)
            # Fill Theta and estimate Phi with their expectation
            if (t_burn+1) % 10 == 0:
                Theta = c_topic + alpha
                Theta = np.transpose(Theta.transpose() / np.sum(Theta, axis=1))
                Phi = c_word + beta
                Phi = np.transpose(Phi.transpose() / np.sum(Phi, axis=1))
                # report the log-likelihood for documents
                print("The log-likelihood at iteration ", t_burn+1, ": ", self.score(data, Theta, Phi))

        # ---sampling from the distribution after burn-in
        for t_sample in range(0, n_sample):
            for m in range(0, M):
                words_m = np.nonzero(data[m])[0]
                for w_id in words_m:
                    w_tp = Z[m, w_id]
                    c_topic[m, w_tp-1] -= 1
                    c_word[w_tp-1, w_id] -= 1
                    # ------select a new topic
                    z_pval = np.array([(c_word[k, w_id]+beta[w_id])*(c_topic[m, k]+alpha[k])/(np.sum(c_word[k, :])+np.sum(beta))
                                       for k in range(0, self.K)])
                    z_pval = z_pval / np.sum(z_pval)
                    tp_new = np.random.choice(topic_set, 1, p=z_pval)[0]
                    Z[m, w_id] = tp_new
                    # ------update the counter matrix
                    c_topic[m, tp_new-1] += 1
                    c_word[tp_new-1, w_id] += 1

        Theta = c_topic + alpha
        Theta = np.transpose(Theta.transpose() / np.sum(Theta, axis=1))
        Phi = c_word + beta
        Phi = np.transpose(Phi.transpose() / np.sum(Phi, axis=1))
        # report the log-likelihood for documents
        print("The final log-likelihood: ", self.score(data, Theta, Phi))

        return Z, Phi

    def score(self, data, Theta, Phi):
        s = 0
        M = len(Theta)
        for m in range(0, M):
            words_m = np.nonzero(data[m])[0]
            for w_id in words_m:
                s += np.log(np.sum([Theta[m, k]*Phi[k, w_id] for k in range(0, self.K)]))
        return s


# construct word frequency vectors from input data
f = open("data.txt")
next(f)
corpus = [line.strip() for line in f]
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
vocabulary = vectorizer.get_feature_names()
word_frequency = X.toarray()  # shape: (10768, 6986)

model = ldaModel(n_topics=int(sys.argv[1]), a=0.5, b=0.5)

#np.random.seed(111)
start = timeit.default_timer()
z_fit, phi_fit = model.fit(data=word_frequency, Vmax=len(vocabulary), n_sample=100, n_burn=4000)
stop = timeit.default_timer()
print("Run-time of Fitting: ", stop - start)
print(" ")
print("Topic Explanation")
for k in range(0, model.K):
    print("********Topic ", k, "********")
    word_max = phi_fit[k, :].argsort()[-10:][::-1]
    for w_id in word_max:
        print(vocabulary[w_id], ": ", phi_fit[k, w_id])
