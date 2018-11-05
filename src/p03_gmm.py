'''
File to explore using a Gaussian mixture model for the project.
Uses code from pset 3 for a quick look at feasibility.
Will need to be generalized and performance improved or use standard library.
Only runs for total reads from f1 at the moment.

Usage (run from project directory):
    python src/p03_gmm.py
'''

import json
import os

import matplotlib.pyplot as plt
import numpy as np


PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 10  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('data', 'f1_reads.json')
    total_reads = load_rend_seq(train_path)

    window = 11
    clipped_index = int((window-1) / 2)  # For proper indexing since data is lost
    ma = np.convolve(total_reads, np.ones((window,))/window, 'valid')

    x = np.vstack((ma, total_reads[clipped_index:- clipped_index], range(len(ma)))).T

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m, n = x.shape
    mu = np.zeros((K, n))
    sigma = np.zeros((K, n, n))

    n_samples = m // K
    idx = np.arange(m)
    np.random.shuffle(idx)

    for i in range(K):
        samples = x[idx[i*n_samples:(i+1)*n_samples], :]
        mu[i, :] = np.mean(samples, axis=0)
        diff = mu[i, :] - samples
        sigma[i, :, :] = np.dot(diff.T, diff) / m

    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.ones(K) / K

    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w = np.ones((m, K)) / K
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plt.figure()
    plt.plot(range(len(z_pred)), z_pred)
    plt.savefig('output/levels_{}.png'.format(trial_num))

    window = 31
    ma = np.convolve(z_pred, np.ones((window,))/window, 'valid')
    plt.figure()
    plt.plot(range(len(ma)), ma)
    plt.savefig('output/levels_ma_{}.png'.format(trial_num))

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE
        m, n = x.shape

        # (1) E-step: Update your estimates in w
        for i, sample in enumerate(x):
            numerator = np.zeros(K)
            for j in range(K):
                numerator[j] = gaussian(sample, mu[j, :], sigma[j, :, :]) * phi[j]
            if np.sum(numerator) == 0:
                w[i, :] = 0
            else:
                w[i, :] = numerator / np.sum(numerator)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        w_sum = np.sum(w, axis=0)
        phi = w_sum / m
        mu = np.dot(w.T, x) / w_sum.reshape(-1, 1)
        for i in range(K):
            diff = x - mu[i, :]
            sigma[i, :, :] = np.dot(diff.T, w[:, i].reshape(-1, 1) * diff) / w_sum[i]

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = 0
        for i, sample in enumerate(x):
            likelihood = np.zeros(K)
            for j in range(K):
                likelihood[j] = gaussian(sample, mu[j, :], sigma[j, :, :]) * phi[j]
            ll += np.log(np.sum(likelihood))

        it += 1

    print('{}: {}'.format(it, ll))
        # *** END CODE HERE ***

    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        # *** START CODE HERE ***
        m, n = x.shape
        m_tilde = x_tilde.shape[0]

        # (1) E-step: Update your estimates in w
        for i, sample in enumerate(x):
            numerator = np.zeros(K)
            for j in range(K):
                numerator[j] = gaussian(sample, mu[j, :], sigma[j, :, :]) * phi[j]
            w[i, :] = numerator / np.sum(numerator)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        wz = np.zeros((m_tilde, K))
        for i, val in enumerate(z):
            wz[i, int(val)] = 1

        w_sum = np.sum(w, axis=0)
        z_sum = np.sum(wz, axis=0)
        phi = (w_sum + alpha * z_sum) / (m + alpha * m_tilde)

        mu = (np.dot(w.T, x) + alpha * np.dot(wz.T, x_tilde)) / (w_sum.reshape(-1, 1) + alpha * z_sum.reshape(-1, 1))

        for i in range(K):
            diff1 = x - mu[i, :]
            diff2 = x_tilde - mu[i, :]
            unsupervised = np.dot(diff1.T, w[:, i].reshape(-1, 1) * diff1)
            supervised = np.dot(diff2.T, wz[:, i].reshape(-1, 1) * diff2)
            sigma[i, :, :] = (unsupervised + alpha * supervised) / (w_sum[i] + alpha * z_sum[i])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        ll = 0
        for sample in x:
            likelihood = np.zeros(K)
            for i in range(K):
                likelihood[i] = gaussian(sample, mu[i, :], sigma[i, :, :]) * phi[i]
            ll += np.log(np.sum(likelihood))
        for sample, group in zip(x_tilde, z):
            i = int(group)
            ll += alpha * np.log(gaussian(sample, mu[i, :], sigma[i, :, :]) * phi[i])

        print('{}: {}'.format(it, ll))
        it += 1
        # *** END CODE HERE ***

    return w


# *** START CODE HERE ***
def gaussian(x, mu, sigma):
    '''
    Returns a probability from a Gaussian distribution given a sample
    x and parameters mu and sigma.

    Args:
        x (array): sample, size n
        mu (array): mean, size n
        sigma (2D array): covariance, size (n x n)
    '''

    diff = x - mu
    exp = -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff)
    return 1 / np.sqrt(2*np.pi*np.linalg.det(sigma)) * np.exp(exp)
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_rend_seq(path):
    '''
    Load saved sequencing data.

    Args:
        path (str): path to file to read

    Returns:
        array (floats): total reads at each location
    '''

    with open(path) as f:
        data = json.load(f)

    return np.array(data)


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
