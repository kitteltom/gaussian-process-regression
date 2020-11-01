# Gaussian Process Regression

import numpy as np
import scipy.optimize


class SquareExponential:
    """
    Square Exponential kernel
    """

    def __init__(self, sigma=1., ell=1.):
        """
        Initializes the square exponential kernel.

        :param float sigma: Gain of the SE-kernel
        :param float ell: Length scale of the SE-kernel
        """
        self.sigma = sigma
        self.ell = ell

    def k(self, A, B):
        """
        Calculate the kernel matrix.

        :param numpy array, shape (N, 1) A: Input matrix
        :param numpy array, shape (N, 1) B: Input matrix
        :return: numpy array, shape (N, N) k: Kernel matrix
        """
        return self.sigma ** 2 * np.exp(-squared_l2_norm(A, B) / (2. * self.ell ** 2))

    def dk_dsigma(self, X):
        """
        Derivative of the kernel matrix with respect to sigma.

        :param numpy array, shape (N, 1) X: Data points
        :return: numpy array, shape (N, N) dk_dsigma: Derivative w.r.t sigma
        """
        return 2 * self.k(X, X) / self.sigma

    def dk_dell(self, X):
        """
        Derivative of the kernel matrix with respect to ell.

        :param numpy array, shape (N, 1) X: Data points
        :return: numpy array, shape (N, N) dk_dell: Derivative w.r.t ell
        """
        return self.k(X, X) * squared_l2_norm(X, X) / self.ell ** 3


class GaussianProcessRegression:
    """
    Gaussian Process Regression with Square Exponential kernel
    """

    def __init__(self, X, Y, noise=0.1, sigma=1., ell=1.):
        """
        Initializes gaussian process regression.

        :param numpy array, shape (N, 1) X: Data points
        :param numpy array, shape (N, 1) Y: Function values
        :param float noise: Measurement noise
        :param float sigma: Gain of the SE-kernel
        :param float ell: Length scale of the SE-kernel
        """
        self.X = X
        self.Y = Y
        self.noise = noise
        self.kernel = SquareExponential(sigma, ell)
        self.L, self.alpha = self.__invert_gram_matrix()
        self.plot_cnt = 0

    def __invert_gram_matrix(self):
        """
        Calculates and inverts the gram matrix with cholesky decomposition.

        :return: numpy array, shape (N, N) L: Lower triangular cholesky factor of the gram matrix
                 numpy array, shape (N, 1) alpha: Inverted gram matrix multiplied with the function values Y
        """
        kXX = self.kernel.k(self.X, self.X)
        gram_matrix = kXX + self.noise ** 2 * np.eye(X.shape[0])
        L = np.linalg.cholesky(gram_matrix)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.Y))
        return L, alpha

    def neg_log_marginal_likelihood(self):
        """
        Calculates the negative logarithm of the marginal likelihood. The marginal likelihood is a multivariate
        gaussian with 0 mean and the gram_matrix as covariance-matrix. (See Rasmussen & Williams, 2006, Section 2.2
        as a reference.)

        :return: float log_ml: Negative log marginal likelihood
        """
        return 0.5 * (self.Y.T @ self.alpha)[0, 0] \
            + np.sum(np.log(np.diag(self.L))) \
            - self.Y.shape[0]/2. * np.log(2 * np.pi)

    def __dml_dtheta(self, theta, plot_steps, plot_data):
        """
        Calculate the derivative of the negative log marginal likelihood w.r.t the parameters of the kernel.
        (See Rasmussen & Williams, 2006, Section 5.4.1 as a reference.)

        :param numpy array, shape (2, 1) theta: Parameters of the kernel
        :param boolean plot_steps: Plot optimization steps if True
        :param plot_data: Unseen data to predict and plot
        :return: float log_ml: Negative log marginal likelihood
                 numpy array, shape (2, 1) dml_dtheta: Derivative w.r.t the kernel parameters
        """
        self.kernel = SquareExponential(theta[0], theta[1])
        self.L, self.alpha = self.__invert_gram_matrix()

        if plot_steps and plot_data is not None:
            self.plot(plot_data, *self.predict(plot_data))

        dml_dtheta = []
        for dk_dtheta in [self.kernel.dk_dsigma(self.X), self.kernel.dk_dell(self.X)]:
            dml_dtheta.append(
                -0.5 * (self.alpha.T @ dk_dtheta @ self.alpha)[0, 0]
                + 0.5 * np.trace(np.linalg.solve(self.L.T, np.linalg.solve(self.L, dk_dtheta)))
            )
        return self.neg_log_marginal_likelihood(), np.array(dml_dtheta)

    def train(self, plot_steps=False, plot_data=None):
        """
        Trains the gaussian process regression by optimizing the negative log marginal likelihood w.r.t the
        parameters of the kernel.

        :param boolean plot_steps: Plot optimization steps if True
        :param plot_data: Unseen data to predict and plot if plot_steps is True
        """
        theta = np.array([self.kernel.sigma, self.kernel.ell])
        res = scipy.optimize.minimize(
            self.__dml_dtheta,
            theta,
            args=(plot_steps, plot_data),
            jac=True,
            options={'maxiter': 25}
        )
        print('Optimized values: sigma = %.3f, ell = %.3f' % (res.x[0], res.x[1]))

    def predict(self, x):
        """
        Predicts function values of unseen data x using gaussian process regression.

        :param numpy array, shape (N, 1) x: Unseen data points
        :return: numpy array, shape (N, 1) mu_post: Point estimates of the function values
                 numpy array, shape (N, N) cov_post: Covariance matrix of the function values
        """
        kxx = self.kernel.k(x, x)
        kXx = self.kernel.k(self.X, x)
        gain = np.linalg.solve(self.L.T, np.linalg.solve(self.L, kXx)).T
        mu_post = gain @ self.Y
        cov_post = kxx - gain @ kXx
        return mu_post, cov_post

    def plot(self, x, mu_post, cov_post):
        """
        Plot the training points, the real function, the predicted function and 95% confidence intervals for the
        predicted function values and save the plot to 'out/'.

        :param numpy array, shape (N, 1) x: Data points used for prediction
        :param numpy array, shape (N, 1) mu_post: Point estimates of the predicted function values
        :param numpy array, shape (N, N) cov_post: Covariance matrix of the predicted function values
        """
        fig = plt.figure(figsize=(8, 4))
        plt.plot(x, f(x), 'k', label='Real function')
        plt.plot(x, mu_post, 'C1', label='Prediction')
        plt.fill_between(
            x.ravel(),
            mu_post.ravel() - 1.96 * np.sqrt(np.diag(cov_post)),
            mu_post.ravel() + 1.96 * np.sqrt(np.diag(cov_post)),
            color='C1',
            alpha=0.3,
            label='95%'
        )
        plt.scatter(self.X, self.Y, alpha=1, marker='o', color='C0', label='Training data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim((-0.2, 1.2))
        plt.ylim((-20.0, 20.0))
        plt.title('Gaussian Process Regression')
        plt.legend(loc='lower left')
        plt.savefig('out/gp_regression_%.3d.png' % self.plot_cnt)
        plt.close(fig)
        self.plot_cnt += 1


def squared_l2_norm(A, B):
    """
    Helper function to calculate the squared l2 norm between each possible combination of the rows of A and B.

    :param numpy array, shape (N, 1) A: Input matrix
    :param numpy array, shape (N, 1) B: Input matrix
    :return: numpy array, shape (N, N) l2_norm: Matrix of l2 norms
    """
    return np.array([[np.sum((A[i, :] - B[j, :]) ** 2)
                      for j in range(B.shape[0])]
                     for i in range(A.shape[0])])


def f(x):
    """
    Example function to test GP regression.

    :param numpy array, shape (N, 1) x: Data points
    :return: numpy array, shape (N, 1) f: Function values
    """
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


# Execute GP regression with a sample dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a sample dataset with random noise
    noise = 0.4
    X = np.linspace(0, 1.0, 10).reshape(-1, 1)
    Y = f(X) + noise * np.random.randn(*X.shape)
    x = np.linspace(-0.2, 1.2, 100).reshape(-1, 1)

    # Fit GP regression to the data
    GP = GaussianProcessRegression(X, Y, noise=noise)
    GP.train(plot_steps=True, plot_data=x)

    # Predict unseen data
    y_pred, cov_pred = GP.predict(x)

    # Plot the data
    GP.plot(x, y_pred, cov_pred)
