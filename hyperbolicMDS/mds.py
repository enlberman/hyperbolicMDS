import numpy
import numpy as np
from sklearn.metrics import euclidean_distances


# define helper functions
def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)


# convert array or list [a, b] to complex number a+bi
def to_complex(xi):
    return np.complex(xi[0], xi[1])


def cart_to_polar(X):
    r = X.shape[0]
    c = X.shape[1]
    Y = np.ones(X.shape)
    Y[:, 0] = np.sqrt((X ** 2).sum(1))
    temp = Y[:, 0]
    if c > 1:
        for k in range(0, c - 2):
            theta = np.arccos(X[:, c - k - 1] / temp)
            Y[:, k + 1] = theta
            temp = temp * np.sin(theta)
        for k in range(0, r):
            if X[k, 1] / temp[k] > 0:
                Y[k, c - 1] = np.arccos(X[k, 0] / temp[k])
            else:
                Y[k, c - 1] = 2 * np.pi - np.arccos(X[k, 0] / temp[k])
    return (Y)


def polar_to_cart(X):
    c = X.shape[1]
    Y = np.ones(X.shape)
    for k in range(c):
        Y[:, k] = X[:, 0]
        for j in range(k + 1):
            if j < k:
                Y[:, k] = Y[:, k] * np.sin(X[:, j + 1])
            elif (j == k) and ((j + 1) < c):
                Y[:, k] = Y[:, k] * np.cos(X[:, j + 1])
    return (Y)


def sample_hyperbolic_space(n, rmin, rmax, dimensions=3):
    x = np.linspace(rmin, rmax, 10000)
    wv = np.sinh((dimensions - 1) * x)
    wv = wv / wv.sum()
    radii = np.random.choice(x, size=n, p=wv)
    thetas = np.random.rand(n, dimensions - 2) * np.pi
    phi = np.random.rand(n, 1) * 2 * np.pi
    angles = np.hstack([thetas, phi])

    X = polar_to_cart(
        np.hstack([
            radii.reshape(-1, 1),
            angles])
    )
    return X


def poincare_dist_vec(z):
    cos_angle = z @ z.T
    z = cart_to_polar(z)
    cos_angle = cos_angle / (z[:, 0].reshape(-1, 1) @ z[:, 0].reshape(-1, 1).T)
    pre = np.cosh(z[:, 0].reshape(-1, 1)) * np.cosh(z[:, 0].reshape(-1, 1).T) - np.sinh(
        z[:, 0].reshape(-1, 1)) * np.sinh(z[:, 0].reshape(-1, 1).T) * cos_angle
    pre[pre < 1] = 1
    return np.arccosh(pre)


def partial_d_vec(x):
    ri = np.sqrt((x ** 2).sum(1)).reshape(-1, 1)
    rj = ri.T
    dot_product = x @ x.T
    n1 = [x[:, k].reshape(-1, 1) * np.ones(x.shape[0]).reshape(-1, 1).T
          * (
                np.sinh(ri) @ np.sinh(rj) * dot_product / (ri ** 3 @ rj)) for k in range(x.shape[1])]
    n2 = [x[:, k].reshape(-1, 1) * np.ones(x.shape[0]).reshape(-1, 1).T *
          (np.cosh(ri) @ np.sinh(rj)) * dot_product / (
                ri ** 2 @ rj) for k in range(x.shape[1])]
    n3 = [(x[:, k].reshape(-1, 1) *
           np.ones(x.shape[0]).reshape(-1, 1).T).T *
          (np.sinh(ri) @ np.sinh(rj) / (ri @ rj))
          for k in range(x.shape[1])]
    n4 = [x[:, k].reshape(-1, 1) *
          np.ones(x.shape[0]).reshape(-1, 1).T *
          (np.sinh(ri) @ np.cosh(rj) / ri) for k in
          range(x.shape[1])]
    d1 = np.sinh(ri) @ np.sinh(rj) * dot_product / (ri @ rj)
    d2 = np.cosh(ri) @ np.cosh(rj)
    n = [n1[k] - n2[k] - n3[k] + n4[k] for k in range(x.shape[1])]
    d = np.sqrt((-d1 + d2 - 1).astype(complex)) * np.sqrt((-d1 + d2 + 1).astype(complex))
    for k in range(x.shape[1]):
        n[k][d == 0] = 0.
    d[d == 0] = 1.
    part = [np.real(n[k] / d) for k in range(x.shape[1])]
    return part


def partial_d_ds_vec(x, ds, grad):
    x_ds = x - grad * ds
    r_ds = np.sqrt((x_ds ** 2).sum(1))
    dot_product = x @ x.T

    g = dot_product / (r_ds.reshape(-1, 1) @ r_ds.reshape(-1, 1).T)
    fp = (2 * grad ** 2 * ds - 2 * grad * x).sum(1) / (2 * r_ds)
    gp1 = np.dstack([-(grad * x_ds)[:, k] * np.ones((x.shape[0], x.shape[0])) - (
                (grad * x_ds)[:, k] * np.ones((x.shape[0], x.shape[0]))).T for k in range(x.shape[1])]).sum(2) / (
                      r_ds.reshape(-1, 1) @ r_ds.reshape(-1, 1).T)
    gp21 = -dot_product / (r_ds.reshape(-1, 1) @ r_ds.reshape(-1, 1).T) ** 2
    gp22 = ((2 * ds * grad ** 2 - 2 * grad * x).sum(1) * r_ds ** 2).reshape(-1, 1) @ (
                (2 * ds * grad ** 2 - 2 * grad * x).sum(1) * r_ds ** 2).reshape(1, -1)
    gp22d = 2 * (r_ds.reshape(-1, 1) @ r_ds.reshape(-1, 1).T)
    gp2 = gp21 * gp22 / gp22d
    gp = gp1 + gp2

    n11 = g * (np.cosh(r_ds).reshape(-1, 1) @ np.sinh(r_ds).reshape(1, -1))
    n12 = (np.sinh(r_ds).reshape(-1, 1) @ np.cosh(r_ds).reshape(1, -1))
    n1 = fp * np.ones((x.shape[0], x.shape[0])) * (n11 + n12)
    n2 = gp * (np.sinh(r_ds).reshape(-1, 1) @ np.sinh(r_ds).reshape(1, -1))
    n31 = g * (np.sinh(r_ds).reshape(-1, 1) @ np.cosh(r_ds).reshape(1, -1))
    n32 = (np.cosh(r_ds).reshape(-1, 1) @ np.sinh(r_ds).reshape(1, -1))
    n3 = (fp * np.ones((x.shape[0], x.shape[0]))).T * (n31 + n32)

    n = n1 + n2 + n3

    d1 = np.sqrt(g * (np.sinh(r_ds).reshape(-1, 1) @ np.sinh(r_ds).reshape(1, -1)) + (
                np.cosh(r_ds).reshape(-1, 1) @ np.cosh(r_ds).reshape(1, -1)) - 1)
    d2 = np.sqrt(g * (np.sinh(r_ds).reshape(-1, 1) @ np.sinh(r_ds).reshape(1, -1)) + (
                np.cosh(r_ds).reshape(-1, 1) @ np.cosh(r_ds).reshape(1, -1)) + 1)
    d = d1 * d2

    part = np.real(n / d)
    return part


def hypeLineSearch(r0, rm, roof, q):
    r = r0
    while r < rm and q(r) < roof(r):
        r = 2 * r
    while r < rm and q(r) > roof(r):
        r = r / 2
    return r


# what is going on in the line search so i can modify to the poincare ball of n dims
def line_search(Z, dissimilarities, g, r0, rmax, verbose=False):
    p = 0.5
    r = r0
    q0 = sammon_stress_vec(Z, dissimilarities)
    qprime_0 = sammon_stress_grad_step(Z, dissimilarities, g, 0)
    roof_fn = lambda r: q0 + p * qprime_0 * r
    rmin = 1e-5
    while r < rmax and sammon_stress_vec(Z - g * r, dissimilarities) < roof_fn(r):
        if verbose:
            print('step size: ', r)
            print('roof fn: ', roof_fn(r))
            # print('step error: ', step_error(r, Z, g, dissimilarities, n))
        r = 2 * r
    while r > rmax or sammon_stress_vec(Z - g * r, dissimilarities) > roof_fn(r):
        if verbose:
            print('step size: ', r)
            print('roof fn: ', roof_fn(r))
            # print('step error: ', step_error(r, Z, g, dissimilarities, n))
        if r < rmin:
            return 2 * r
        r = r / 2
    return r


class HyperMDS():

    def __init__(self, verbose=0, eps=1e-5, alpha=1, save_metrics=False,
                 random_state=None, dissimilarity="euclidean", dims=3, start_embed=None,stop_idx=5000,check_idx=5000):
        self.dissimilarity = dissimilarity
        self.alpha = alpha
        self.eps = eps
        self.dims = dims
        self.verbose = verbose
        self.random_state = random_state
        self.save_metrics = save_metrics
        self.start_embed = start_embed
        self.stop_idx = stop_idx
        self.check_idx = check_idx
        if self.save_metrics:
            self.gradient_norms = []
            self.steps = []
            self.rel_steps = []
            self.losses = []
            self.best_embedding = None

    def loss_fn(self):
        self.loss = sammon_stress_vec(self.embedding, self.dissimilarity_matrix, alpha=self.alpha)

    def compute_gradients(self):
        self.gradients = sammon_stress_grad_vec(self.embedding, self.dissimilarity_matrix, alpha=self.alpha)

    def fit(self, X):
        """
        Uses gradient descent to find the embedding configuration in the Poincaré disk
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        """
        self.fit_transform(X)
        return self

    def init_embed(self, rmax, rmin):
        random_config = np.random.uniform(-rmax, rmax, size=(self.n, self.dims))
        r = np.random.uniform(rmin, rmax, size=self.n)
        random_config = (random_config.T / (random_config ** 2).sum(1) * r).T
        self.embedding = random_config

    def fit_transform(self, X, rmin=0.9, rmax=4,
                      max_epochs=40, verbose=False,
                      alpha0=1,
                      beta10=.5,
                      beta20=.8,
                      eps=5,
                      tol=1e-5):
        """
        Fit the embedding from X, and return the embedding coordinates
        Parameters
        ----------
        X : array, shape (n_samples, n_features) or (n_samples, n_samples)
            Input data. If ``dissimilarity=='precomputed'``, the input should
            be the dissimilarity matrix.
        init: initial configuration of the embedding coordinates
        init_low: lower bound for range of initial configuration
        init_high: upper bound for range of initial configuration
        smax: max distance window to seek embedding
        max_epochs: maximum number of gradient descent iterations
        verbose: optionally print training scores
        """
        if self.dissimilarity == "precomputed":
            self.dissimilarity_matrix = X
        elif self.dissimilarity == "euclidean":
            self.dissimilarity_matrix = euclidean_distances(X)
        elif self.dissimilarity == 'hyperbolic':
            self.dissimilarity_matrix = poincare_dist_vec(X)
        self.n = self.dissimilarity_matrix.shape[0]

        # set initial config
        if self.start_embed is None:
            self.init_embed(rmax, rmin)
        else:
            self.embedding = self.start_embed

        # set a max distance window for embedding
        prev_loss = np.inf
        m_prev0 = 0
        v_prev0 = 0
        alpha = alpha0
        beta1 = beta10
        beta2 = beta20
        m_prev = m_prev0
        v_prev = v_prev0
        losses = []
        for i in range(max_epochs):
            # break if loss decrease < tolerance
            self.loss_fn()
            losses.append(self.loss)

            if i > self.check_idx:
                if numpy.diff(losses[-self.stop_idx:]).mean() > 0:
                    break
            if i > 50:
                chng = numpy.abs(numpy.diff(losses[-50:]))
                chng = chng.mean()
                if chng < tol:
                    break
            prev_loss = self.loss
            self.compute_gradients()

            beta1 = beta1 ** (np.sqrt(i + 1))
            beta2 = beta2 ** (np.sqrt(i + 1))
            alpha = alpha * np.sqrt(1 - beta2) / (1 - beta1)
            m = einstein_multiplication(self.gradients,beta1 * m_prev + (1 - beta1), rmax)
            v = einstein_multiplication(self.gradients ** 2, beta2 * v_prev + (1 - beta2), rmax)
            m_prev = m
            v_prev = v
            gn = norm(self.gradients, axis=1).max()
            delta = einstein_multiplication(m / (np.sqrt(v) + eps * gn), alpha, rmax)
            tmp = mobius_addition(self.embedding, - delta, rmax)
            tmp_norm = norm(tmp, 1)
            tmp[tmp_norm < rmin, :] = (tmp[tmp_norm < rmin, :].T / tmp_norm[tmp_norm < rmin] * rmin).T
            tmp[tmp_norm > rmax, :] = (tmp[tmp_norm > rmax, :].T / tmp_norm[tmp_norm > rmax] * rmax).T
            self.embedding = tmp

            # # optionally save training metrics
            if self.save_metrics:
                self.gradient_norms.append(norm(self.gradients, axis=1).max())
                # self.steps.append(r)
                # self.rel_steps.append(r / rmax)
                if self.loss == min(losses):
                    self.best_embedding = self.embedding
                self.losses.append(self.loss)

            if verbose & ((i % 50) == 0):
                print('Epoch ' + str(i + 1) + ' complete')
                print('Loss: ', self.loss)
                print('\n')

        self.loss_fn()

        return self.embedding


# ----------------------------------------
# ----- EVALUATION UTILITY FUNCTIONS -----
# ----------------------------------------

def sammon_stress_vec(embedding, dissimilarity_matrix, alpha=1):
    d = poincare_dist_vec(embedding)
    delta = alpha * dissimilarity_matrix
    deltaz = delta.copy()
    deltaz[deltaz == 0] = 1.
    return np.triu((d - delta) ** 2 / deltaz).sum() / np.triu(delta).sum()


# compute gradient of Sammon stress of the embedding
def sammon_stress_grad_vec(embedding, dissimilarity_matrix, alpha=1):
    pd = partial_d_vec(embedding)
    d = poincare_dist_vec(embedding)
    delta = alpha * dissimilarity_matrix
    deltaz = delta.copy()
    deltaz[deltaz == 0] = 1.
    grad = np.vstack(
        [np.triu(2 * (d - delta) * pd[k] / deltaz).sum(1) / np.triu(delta).sum() for k in range(embedding.shape[1])]).T
    return grad


# compute gradient of Sammon stress of the embedding with respect to some step size
def sammon_stress_grad_step(embedding, dissimilarity_matrix, g, ds, alpha=1):
    scale = 0
    n = embedding.shape[0]
    pd_step = partial_d_ds_vec(embedding, ds, g)
    pd = poincare_dist_vec(embedding - g * ds)
    grad = 0
    for i in range(n):
        for j in range(i + 1, n):
            if dissimilarity_matrix[i][j] != 0:
                delta_ij = alpha * dissimilarity_matrix[i][j]
                d_ij = pd[i, j]
                d_ij_p = pd_step[i, j]
                num = 2 * ds * d_ij * d_ij_p - 2 * delta_ij * d_ij_p
                scale += delta_ij
                grad += num / delta_ij
    return grad / scale


import numpy as np


def mobius_addition(u, v, s):
    # left addition of u to v with max hyperbolic radius s
    u_norm = np.sqrt((u**2).sum(1))
    v_norm = np.sqrt((v ** 2).sum(1))
    u_dot_v = (u * v).sum(axis=1)
    numerator = (((1 +2/s**2 * u_dot_v + v_norm**2/s**2)*u.T).T + ((1-u_norm**2/s**2)*v.T).T)
    denominator = (1 +2/s**2 * u_dot_v + v_norm**2/s**4*u_norm**2)
    return (numerator.T/denominator).T


def einstein_multiplication(u,r,s):
    u_norm = np.sqrt((u ** 2).sum(1))
    return ((s * np.tanh(r * np.arctanh(u_norm / s)) * u.T)).T / u

