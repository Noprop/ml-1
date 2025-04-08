import numpy as np


def univariate_loss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    b = theta[0]
    w = theta[1]
    mSq = 0
    for i in range(len(x)):
        newX = b + w*x[i]
        newArray = np.subtract(newX,y)
        mSq += newArray[i]**2

    """
    :param x: 1D array that represents the feature vector
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector theta = (b, w)
    :return: a scalar that represents the loss \mathcal{L}_U(theta)
    """
    # TODO: Implement the univariate loss \mathcal{L}_U(theta) (as specified in Equation 1)
    return mSq


def fit_univariate_lin_model(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_U(theta)
    """

    N = x.size
    assert N > 1, "There must be at least 2 points given!"
    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.1.1)
    xBar= np.average(x)
    yBar = np.average(y)
    a = xBar ** 2
    bx = xBar * yBar
    c = 0
    d = 0
    for i in range(len(x)):
       c += x[i]**2
       d += x[i]*y[i]

    w = (-bx + d)/(-a+c)
    b = -w*xBar + yBar

    res = 0
    for i in range(len(x)):
      res += (b + w * x[i] - y[i])**2

    return np.array([b, w])


def calculate_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
    :param x: 1D array that contains the feature of each subject
    :param y: 1D array that contains the target of each subject
    :return: a scalar that represents the Pearson correlation coefficient between x and y
    """
    xBar= np.average(x)
    yBar = np.average(y)
    xDiff = 0
    yDiff = 0
    for i in range(len(x)):
        xDiff += (x[i]-xBar)
        yDiff += (y[i]-yBar)
        scalar = (xDiff * yDiff)/(((xDiff)**2)**(1/2) * ((yDiff)**2)**(1/2))

    return scalar


def compute_design_matrix(data: np.ndarray) -> np.ndarray:
    """
    :param data: 2D array of shape (N, D) that represents the data matrix
    :return: 2D array that represents the design matrix. Think about the shape of the output.
    """

    # TODO: Implement the design matrix for multiple linear regression (Task 1.2.2)
    design_matrix = None
    return design_matrix


def multiple_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :param theta: 1D array that represents the parameter vector
    :return: a scalar that represents the loss \mathcal{L}_M(theta)
    """
    # TODO: Implement the multiple regression loss \mathcal{L}_M(theta) (as specified in Equation 5)
    return None


def fit_multiple_lin_model(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    :param X: 2D array that represents the design matrix
    :param y: 1D array that represents the target vector
    :return: the parameter vector theta^* that minimizes the loss \mathcal{L}_M(theta)
    """
    from numpy.linalg import pinv

    # TODO: Implement the expressions you have derived in the pen & paper exercise (Task 1.2.1). 
    # Note: Use the pinv function.
    theta = None
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