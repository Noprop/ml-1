import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from plot_utils import (plot_scatterplot_and_line, plot_scatterplot_and_polynomial, 
                        plot_logistic_regression, plot_datapoints, plot_3d_surface, plot_2d_contour,
                        plot_function_over_iterations)
from linear_regression import (fit_univariate_lin_model, 
                               fit_multiple_lin_model, 
                               univariate_loss, multiple_loss,
                               calculate_pearson_correlation,
                               compute_design_matrix,
                               compute_polynomial_design_matrix)
from logistic_regression import (create_design_matrix_dataset_1,
                                 create_design_matrix_dataset_2,
                                 create_design_matrix_dataset_3,
                                 logistic_regression_params_sklearn)
from gradient_descent import rastrigin, gradient_rastrigin, gradient_descent, finite_difference_gradient_approx


def task_1(use_linalg_formulation=False):
    print('---- Task 1 ----')
    column_to_id = {"hours_sleep": 0, "hours_work": 1,
                    "avg_pulse": 2, "max_pulse": 3, 
                    "duration": 4, "exercise_intensity": 5, 
                    "fitness_level": 6, "calories": 7}

    # After loading the data, you can for example access it like this: 
    # `smartwatch_data[:, column_to_id['hours_sleep']]`
    smartwatch_data = np.load('./data/smartwatch_data.npy') 
    
    # TODO: Implement Task 1.1.2: Find 3 pairs of features that have a linear relationship.
    # For each pair, fit a univariate linear regression model: If ``use_linalg_formulation`` is False,
    # call `fit_univariate_lin_model`, otherwise use the linalg formulation in `fit_multiple_lin_model` (Task 1.2.2).
    # For each pair, also calculate and report the Pearson correlation coefficient, the theta vector you found, 
    # the MSE, and plot the data points together with the linear function.
    # Repeat the process for 3 pairs of features that do not have a meaningful linear relationship.
    # print(fit_univariate_lin_model(np.array([1, 3, 5]), np.array([2, 2, 2])))

    # good corr #1
    average_pulse = smartwatch_data[:, 2]
    max_pulse = smartwatch_data[:, 3]
    X = np.column_stack([np.ones(len(average_pulse)), average_pulse]) # lol, multiple ways to create a design matrix
    b, w = fit_univariate_lin_model(average_pulse, max_pulse) if not use_linalg_formulation else fit_multiple_lin_model(X, max_pulse)
    plot_scatterplot_and_line(average_pulse, max_pulse, (b, w), "Average Pulse", "Max Pulse", "Average vs Max Pulse")
    pcc = calculate_pearson_correlation(average_pulse, max_pulse)
    theta = fit_univariate_lin_model (average_pulse, max_pulse) 
    mse = univariate_loss(average_pulse, max_pulse,(b,w))
    print("Good #1:")
    print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")

    # good corr #2
    exercise_duration = smartwatch_data[:, 4]
    fitness_level = smartwatch_data[:, 6]
    X = np.column_stack([np.ones(len(exercise_duration)), exercise_duration])
    b, w = fit_univariate_lin_model(exercise_duration, fitness_level) if not use_linalg_formulation else fit_multiple_lin_model(X, fitness_level)
    plot_scatterplot_and_line(exercise_duration, fitness_level, (b, w), "Exercise Duration",
     "Fitness Level", "Exercise Duration vs Fitness Level")
    pcc = calculate_pearson_correlation(exercise_duration, fitness_level)
    theta = fit_univariate_lin_model (exercise_duration, fitness_level) 
    mse = univariate_loss(exercise_duration, fitness_level, (b,w))
    print("Good #2:")
    print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")

    # good corr #3
    fitness_level = smartwatch_data[:, 6]
    calories = smartwatch_data[:, 7]
    X = np.column_stack([np.ones(len(fitness_level)), fitness_level])
    b, w = fit_univariate_lin_model(fitness_level, calories) if not use_linalg_formulation else fit_multiple_lin_model(X, calories)
    plot_scatterplot_and_line(fitness_level, calories, (b, w), "Fitness Level", "Calories Burned", "Fitness Level vs Calories Burned")
    pcc = calculate_pearson_correlation(fitness_level, calories)
    theta = fit_univariate_lin_model (fitness_level, calories)
    mse = univariate_loss(fitness_level, calories, (b, w))
    print("Good #3:")
    print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")
   
    # poor corr #1
    hours_sleep = smartwatch_data[:, 0]
    hours_work = smartwatch_data[:, 1]
    X = np.column_stack([np.ones(len(hours_sleep)), hours_sleep])
    b, w = fit_univariate_lin_model(hours_sleep, hours_work) if not use_linalg_formulation else fit_multiple_lin_model(X, hours_work)
    plot_scatterplot_and_line(hours_sleep, hours_work, (b, w), "Hours Slept", "Hours Worked", "Hours Slept vs Worked")
    pcc = calculate_pearson_correlation(hours_sleep, hours_work)
    theta = fit_univariate_lin_model (hours_sleep, hours_work)
    mse = univariate_loss(hours_sleep, hours_work,(b,w))
    print("Bad #1:")
    print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")

    # bad corr #2
    hours_sleep = smartwatch_data[:, 0]
    calories = smartwatch_data[:, 7]
    X = np.column_stack([np.ones(len(hours_sleep)), hours_sleep])
    b, w = fit_univariate_lin_model(hours_sleep, calories) if not use_linalg_formulation else fit_multiple_lin_model(X, calories)
    plot_scatterplot_and_line(hours_sleep, calories, (b, w), "Hours of Sleep", "Calories Burned", "Hours of Sleep vs Calories Burned")
    pcc = calculate_pearson_correlation(hours_sleep, calories)
    theta = fit_univariate_lin_model (hours_sleep, calories)
    mse = univariate_loss(hours_sleep, calories, (b,w))
    print("Bad #2:")
    print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")

    # wack corr #3
    hours_work = smartwatch_data[:, 1]
    max_pulse = smartwatch_data[:, 3]
    X = np.column_stack([np.ones(len(hours_work)), hours_work])
    b, w = fit_univariate_lin_model(hours_work, max_pulse) if not use_linalg_formulation else fit_multiple_lin_model(X, max_pulse)
    plot_scatterplot_and_line(hours_work, max_pulse, (b, w), "Hours of Work","Max Pulse", "Hours of Work vs Max Pulse")
    pcc = calculate_pearson_correlation(hours_sleep, max_pulse)
    theta = fit_univariate_lin_model (hours_sleep, max_pulse)
    mse = univariate_loss(hours_sleep, max_pulse, (b, w))
    print("Bad #3:")
    print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")

    # TODO: Implement Task 1.2.3: Multiple linear regression
    # Select two additional features, compute the design matrix, and fit the multiple linear regression model.
    # Report the MSE and the theta vector.

    average_pulse = smartwatch_data[:, 2]
    exercise_duration = smartwatch_data[:, 6]
    fitness_level = smartwatch_data[:, 4]
    max_pulse = smartwatch_data[:, 3]
    X = np.column_stack([np.ones(len(average_pulse)), average_pulse, exercise_duration, fitness_level])
    theta = fit_multiple_lin_model(X, max_pulse)
    print(f"b: {theta[0]:.2f} w1: {theta[1]:.2f} w2: {theta[2]:.2f} w3: {theta[3]:.2f}")
    mse = multiple_loss(X, max_pulse, theta)
    print(f"mse: {mse:.2f}")

    # plot_scatterplot_and_line(average_pulse, max_pulse, (b, w), "Average Pulse", "Max Pulse", "Average vs Max Pulse")
    # pcc = calculate_pearson_correlation(average_pulse, max_pulse)
    # theta = fit_univariate_lin_model (average_pulse, max_pulse) 
    # mse = univariate_loss(average_pulse, max_pulse,(b,w))
    # print("Good #1:")
    # print(f"PCC: {pcc:.2f}, b: {theta[0]:.2f}, w: {theta[1]:.2f}, mse: {mse:.2f}")

    # TODO: Implement Task 1.3.2: Polynomial regression
    # For the feature-target pair of choice, compute the polynomial design matrix with an appropriate degree K, 
    # fit the model, and plot the data points together with the polynomial function.
    # Report the MSE and the theta vector.
    average_pulse = smartwatch_data[:, 2]
    max_pulse = smartwatch_data[:, 3]

    X_poly = compute_polynomial_design_matrix(average_pulse, 10)
    theta_poly = np.linalg.lstsq(X_poly, max_pulse, rcond=None)[0]
    y_pred = X_poly.dot(theta_poly)
    mse = np.mean((max_pulse - y_pred) ** 2)
    print("\nPolynomial Regression (Task 1.3.2):")
    print(f"Feature: Hours Sleep vs Target: Average Pulse")
    print(f"Theta: {theta_poly}")
    print(f"MSE: {mse:.2f}")

    plot_scatterplot_and_polynomial(average_pulse, max_pulse, theta_poly, 10, "Average Pulse", "Max Pulse",    
                                    "Average Pulse vs Max Pulse")

    # TODO: Implement Task 1.3.3: Use x_small and y_small to fit a polynomial model.
    # Find and report the smallest K that gets zero loss. Plot the data points and the polynomial function.
    x_small = smartwatch_data[:5, column_to_id['duration']]
    y_small = smartwatch_data[:5, column_to_id['calories']]

    X_poly_small = compute_polynomial_design_matrix(x_small, 4)
    theta_poly_small = np.linalg.lstsq(X_poly_small, y_small, rcond=None)[0]
    y_pred_small = X_poly_small.dot(theta_poly_small)
    mse_small = np.mean((y_small - y_pred_small) ** 2)

    print("\nPolynomial Regression (Task 1.3.3):")
    print(f"Using the small dataset (first 5 samples of 'duration' vs 'calories'):")
    print(f"Smallest polynomial degree that achieves zero loss: {4}")
    print(f"Theta: {theta_poly_small}")
    print(f"MSE: {mse_small:.2e}")
    plot_scatterplot_and_polynomial(x_small, y_small, theta_poly_small, 4,
                                    "Duration", "Calories",
                                    "Polynomial Regression on small dataset: Duration vs Calories")


def task_2():
    print('\n---- Task 2 ----')

    for task in [1, 2, 3]:
        print(f'---- Logistic regression task {task} ----')
        if task == 1:
            # Load the data set 1 (X-1-data.npy and targets-dataset-1.npy)
            X_data = np.load("./data/X-1-data.npy")
            y = np.load("./data/targets-dataset-1.npy")
            create_design_matrix = create_design_matrix_dataset_1
        elif task == 2:
            # Load the data set 2 (X-1-data.npy and targets-dataset-2.npy)
            X_data = np.load("./data/X-1-data.npy")
            y = np.load("./data/targets-dataset-2.npy")
            create_design_matrix = create_design_matrix_dataset_2
        elif task == 3:
            # Load the data set 3 (X-2-data.npy and targets-dataset-3.npy)
            X_data = np.load("./data/X-2-data.npy")
            y = np.load("./data/targets-dataset-3.npy")
            create_design_matrix = create_design_matrix_dataset_3
        else:
            raise ValueError('Task not found.')

        X = create_design_matrix(X_data)

        # Plot the datapoints (just for visual inspection)
        plot_datapoints(X, y, f'Targets - Task {task}')

        # 2.2
        # Split the dataset using the `train_test_split` function.
        # The parameter `random_state` should be set to 0.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        print(f'Shapes of: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}')

        # 2.3
        # Train the classifier
        custom_params = logistic_regression_params_sklearn()
        clf = LogisticRegression(**custom_params)
        # Fit the model to the data using the `fit` method of the classifier `clf`
        clf.fit(X_train, y_train)
        acc_train, acc_test = clf.score(X_train, y_train), clf.score(X_test, y_test)

        print(f'Train accuracy: {acc_train * 100:.2f}%. Test accuracy: {100 * acc_test:.2f}%.')
        
        yhat_train = clf.predict_proba(X_train)[:, 1]
        yhat_test  = clf.predict_proba(X_test)[:, 1]

        #  Use the `log_loss` function to calculate the cross-entropy loss
        #  (once on the training set, once on the test set).
        #  You need to pass (1) the true binary labels and (2) the probability of the *positive* class to `log_loss`.
        #  Since the output of `predict_proba` is of shape (n_samples, n_classes), you need to select the probabilities
        #  of the positive class by indexing the second column (index 1).
        loss_train = log_loss(y_train, yhat_train)
        loss_test  = log_loss(y_test, yhat_test)

        print(f'Train loss: {loss_train:.2f}. Test loss: {loss_test:.2f}.')

        plot_logistic_regression(clf, create_design_matrix, X_train, f'(Dataset {task}) Train set predictions',
                                 figname=f'logreg_train{task}')
        plot_logistic_regression(clf, create_design_matrix, X_test,  f'(Dataset {task}) Test set predictions',
                                 figname=f'logreg_test{task}')

        classifier_weights = clf.coef_[0]
        classifier_bias = clf.intercept_[0]

        print(f'Parameters: {classifier_weights.round(2)}, {classifier_bias:.2f}')


def task_3(initial_plot=True):
    print('\n---- Task 3 ----')
    # Do *not* change this seed
    np.random.seed(46)

    # TODO: Choose a random starting point using samples from a standard normal distribution
    x0 = np.random.randn()
    y0 = np.random.randn()
    print(f'Starting point: {x0:.4f}, {y0:.4f}')
    
    if initial_plot:
        # Plot the function to see how it looks like
        plot_3d_surface(rastrigin)
        plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0))

    # TODO: Check if gradient_rastrigin is correct at (x0, y0). 
    # To do this, print the true gradient and the numerical approximation.
    pass
    
    analytical_grad = gradient_rastrigin(x0, y0)
    numerical_grad = finite_difference_gradient_approx(rastrigin, x0, y0)
    print(f"Analytical gradient at starting point: {analytical_grad}")
    print(f"Numerical gradient at starting point: {numerical_grad}")

    # TODO: Call the function `gradient_descent` with a chosen configuration of hyperparameters,
    #  i.e., learning_rate, lr_decay, and num_iters. Try out lr_decay=1 as well as values for lr_decay that are < 1.
    x_list, y_list, f_list = None, None, None
    lr = 0.02
    # lr_decay = 0.99
    lr_decay = 1
    num_iters = 200
    x_list, y_list, f_list = gradient_descent(rastrigin, gradient_rastrigin, x0, y0, lr, lr_decay, num_iters)

    # Print the point that is found after `num_iters` iterations
    print(f'Solution found: f({x_list[-1]:.4f}, {y_list[-1]:.4f})= {f_list[-1]:.4f}' )
    print(f'Global optimum: f(0, 0)= {rastrigin(0, 0):.4f}')

    # Here we plot the contour of the function with the path taken by the gradient descent algorithm
    plot_2d_contour(rastrigin, starting_point=(x0, y0), global_min=(0, 0), 
                    x_list=x_list, y_list=y_list)

    # TODO: Create a plot f(x_t, y_t) over iterations t by calling `plot_function_over_iterations` with `f_list`
    print(f_list)
    plot_function_over_iterations(f_list)

    pass


def main():
    np.random.seed(46)
    # print("test")

    # task_1(use_linalg_formulation=False)
    # task_2()
    task_3(initial_plot=True)


if __name__ == '__main__':
    main()
