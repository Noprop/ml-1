import numpy as np


def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)

    b = theta[0]
    w = theta[1]

    return np.sum(((b + w*x) - y)**2)/x.size


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """
    N = x.size
    assert N > 1, "There must be at least 2 points given!"

    x_bar = np.mean(x)
    y_bar = np.mean(y)

    num = np.sum(x * y) - y_bar * np.sum(x)
    denom = np.sum(x**2) - x_bar * np.sum(x)
    
    w = num / denom
    b = y_bar - w * x_bar
    
    return np.array([b, w])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    x_bar = np.mean(x)
    y_bar = np.mean(y)

    num = np.sum((x - x_bar) * (y - y_bar))
    denom = np.sqrt(np.sum((x - x_bar)**2)) * np.sqrt(np.sum((y - y_bar)**2))

    return num / denom


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """
    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)

    design_matrix = np.insert(data, 0, 1, axis=1)
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)

    return np.sum(((X @ theta) - y)**2)


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.2.1). 
    # Note: Use the pinv function.

    theta = pinv(X) @ y
    return theta


def compute_polynomial_design_matrix(x: np.ndarray, K: int) -> np.ndarray:
    """
    :param x: 1D array that represents the feature vector
    :param K: the degree of the polynomial
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the polynomial design matrix (Task 1.3.2)
    polynomial_design_matrix = None
    return polynomial_design_matrix