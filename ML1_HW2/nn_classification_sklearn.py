from typing import Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import warnings
# We will suppress ConvergenceWarnings in this task. In practice, you should take warnings more seriously.
warnings.filterwarnings("ignore")


def reduce_dimension(X_train: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    :param X_train: Training data to reduce the dimensionality. Shape: (n_samples, n_features)
    :param n_components: Number of principal components
    :return: Data with reduced dimensionality, which has shape (n_samples, n_components), and the PCA object
    """

    # TODO: Create a PCA object and fit it using X_train
    #       Transform X_train using the PCA object.
    #       Print the explained variance ratio of the PCA object.
    #       Return both the transformed data and the PCA object.

    pca_OBj = PCA(n_components=n_components, random_state=42)
    pca_OBj.fit(X_train)
    new_XT = pca_OBj.fit_transform(X_train)

    return new_XT, pca_OBj


def train_nn(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier with different number of neurons and hidden layers.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    # TODO: Train MLPClassifier with different number of layers/neurons.
    # Print the train accuracy, validation accuracy, and the training loss for each configuration.
    # Return the MLPClassifier that you consider to be the best.

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)
    mlp_best = None
    mlp_best_loss = float('inf')

    hidden_layers = [(2,),(8,),(64,),(256,),(1024,),(128,256,128)]
    for i in range(len(hidden_layers)):
      mlp_c = MLPClassifier(max_iter=100, solver="adam", random_state=1, hidden_layer_sizes=hidden_layers[i])
      mlp_c.fit(X_train, y_train)

      print("Validation Scores for layer size", hidden_layers[i])
      print("Best Loss: ", mlp_c.best_loss_)
      print("Final Loss: ", mlp_c.loss_)
      print("Training accuracy:", mlp_c.score(X_train, y_train))
      print("Test set accuracy:", mlp_c.score(X_val, y_val))
      print()
      if (mlp_c.best_loss_ < mlp_best_loss):
        mlp_best = mlp_c
        mlp_best_loss = mlp_c.best_loss_

    return mlp_best


def train_nn_with_regularization(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Train MLPClassifier using regularization.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The MLPClassifier you consider to be the best
    """
    # TODO: Use the code from the `train_nn` function, but add regularization to the MLPClassifier.
    # Again, return the MLPClassifier that you consider to be the best.

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.2, random_state=42)
    mlp_best = None
    mlp_best_acc = 0

    hidden_layers = [(2,),(8,),(64,),(256,),(1024,),(128,256,128)]
    print("CASE (a)")
    for i in range(len(hidden_layers)):
      mlp_c = MLPClassifier(alpha=0.1, max_iter=100, solver="adam", random_state=1, hidden_layer_sizes=hidden_layers[i])
      mlp_c.fit(X_train, y_train)
      test_acc = mlp_c.score(X_val, y_val)

      print("Validation Scores for layer size", hidden_layers[i])
      print("Best Loss: ", mlp_c.best_loss_)
      print("Final Loss: ", mlp_c.loss_)
      print("Training accuracy:", mlp_c.score(X_train, y_train))
      print("Test set accuracy:", test_acc)
      print()
      if (test_acc > mlp_best_acc):
        mlp_best = mlp_c
        mlp_best_acc = test_acc

    print("\n\nCASE (b)")
    for i in range(len(hidden_layers)):
      mlp_c = MLPClassifier(early_stopping=True, max_iter=100, solver="adam", random_state=1, hidden_layer_sizes=hidden_layers[i])
      mlp_c.fit(X_train, y_train)
      test_acc = mlp_c.score(X_val, y_val)

      print("Validation Scores for layer size", hidden_layers[i])
      print("Best Loss: ", mlp_c.best_loss_)
      print("Final Loss: ", mlp_c.loss_)
      print("Training accuracy:", mlp_c.score(X_train, y_train))
      print("Test set accuracy:", test_acc)
      print()
      if (test_acc > mlp_best_acc):
        mlp_best = mlp_c
        mlp_best_acc = test_acc

    print("\n\nCASE (c)")
    for i in range(len(hidden_layers)):
      mlp_c = MLPClassifier(alpha=0.1, early_stopping=True, max_iter=100, solver="adam", random_state=1, hidden_layer_sizes=hidden_layers[i])
      mlp_c.fit(X_train, y_train)
      test_acc = mlp_c.score(X_val, y_val)

      print("Validation Scores for layer size", hidden_layers[i])
      print("Best Loss: ", mlp_c.best_loss_)
      print("Final Loss: ", mlp_c.loss_)
      print("Training accuracy:", mlp_c.score(X_train, y_train))
      print("Test set accuracy:", test_acc)
      print()
      if (test_acc > mlp_best_acc):
        mlp_best = mlp_c
        mlp_best_acc = test_acc

    return mlp_best

def plot_training_loss_curve(nn: MLPClassifier) -> None:
    """
    Plot the training loss curve.

    :param nn: The trained MLPClassifier
    """
    plt.figure(figsize=(10, 6))
    plt.plot(nn.loss_curve_, 'r', alpha=0.5, label='Loss')
    plt.title('Classifier Loss')
    plt.xlabel('Loss Iteration')
    plt.ylabel('Loss')
    plt.show()


def show_confusion_matrix_and_classification_report(nn: MLPClassifier, X_test: np.ndarray, y_test: np.ndarray) -> None:
    """
    Plot confusion matrix and print classification report.

    :param nn: The trained MLPClassifier you want to evaluate
    :param X_test: Test features (PCA-projected)
    :param y_test: Test targets
    """
    # TODO: Use `nn` to compute predictions on `X_test`.
    #       Use `confusion_matrix` and `ConfusionMatrixDisplay` to plot the confusion matrix on the test data.
    #       Use `classification_report` to print the classification report.

    print("NN score: ", nn.score(X_test, y_test))

    predictions = nn.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    print(classification_report(y_test, predictions))


def perform_grid_search(X_train: np.ndarray, y_train: np.ndarray) -> MLPClassifier:
    """
    Perform GridSearch using GridSearchCV.

    :param X_train: PCA-projected features with shape (n_samples, n_components)
    :param y_train: Targets
    :return: The best estimator (MLPClassifier) found by GridSearchCV
    """
    # TODO: Create parameter dictionary for GridSearchCV, as specified in the assignment sheet.
    #       Create an MLPClassifier with the specified default values.
    #       Run the grid search with `cv=5` and (optionally) `verbose=4`.
    #       Print the best score (mean cross validation score) and the best parameter set.
    #       Return the best estimator found by GridSearchCV.

    griDict = [{
      "alpha": [0,0.1,1],
      "batch_size": [32, 512],
      "hidden_layer_sizes": [(128,),(256,)]
    }]

    mlp_c = MLPClassifier(alpha=0.1, max_iter=100, solver="adam", random_state=1).fit(X_train,y_train)
    gs_Obj = GridSearchCV(estimator=mlp_c, param_grid=griDict, cv=5, verbose=4).fit(X_train,y_train)

    print("Best score: ", gs_Obj.best_score_)
    print("Best Parameter set: ", gs_Obj.best_params_)
    
    return gs_Obj.best_estimator_
