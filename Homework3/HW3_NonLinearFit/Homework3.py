import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor


def round_to_sf(x, sf):
    if x == 0:
        return 0
    return round(x, sf - int(floor(log10(abs(x)))) - 1)

def compute_loss(x, y, n, a, m, b):
    """
    Compute the Mean Squared Error (MSE) loss.

    Parameters:
    x : np.array
        Input data points (x values).
    y : np.array
        Actual output data points (y values).
    n, a, m, b : float
        Parameters of the function y = n * exp(-a * (m * x + b)^2).

    Returns:
    float
        Mean Squared Error (MSE) loss.
    """
    y_int = (m * x + b) ** 2
    y_pred = n * np.exp(-a * y_int)
    return np.mean((y - y_pred) ** 2)


def NonLinearPlotting(x_generated, n_fit, a_fit, m_fit, b_fit, epochs, learning_rate, loss):
    # Predicted y values using the fitted parameters
    y_predicted = n_fit * np.exp(-a_fit * (m_fit * x_generated + b_fit) ** 2)

    plt.figure(1)
    plt.clf()
    # Plot the training data
    plt.scatter(x_generated, y_generated, color='blue', label='Training Data (Noisy)', marker='o')

    # Plot the predicted data
    plt.plot(x_generated, y_predicted, color='red', label='Predicted Data (Model)', linestyle='--')

    # Add labels, title, and legend
    plt.xlabel("x")
    plt.ylabel("y")
    fig_title = "Training vs. predicted data for " + str(epochs) + " epochs and " + str(round_to_sf(learning_rate, 3)) + " learning rate"
    plot_title = str(epochs) + "epochs" + str(round_to_sf(learning_rate, 3)) + "learning rate" + str(round(loss,6)) + "loss"
    plt.title(fig_title)
    plt.legend()
    plt.savefig('HW3_NonLinearFit/Plots/' + str(plot_title) + ".png")
    plt.grid(True)

    # Show the plot
    #plt.show()


def LossPlotting(data_type, ep_plot, rate_plot, loss):

    plt.figure(2)
    plt.clf()
    for ii in range(np.size(loss, 1)):
        plt.plot(ep_plot, loss[:,ii], label='Learning rate: ' + str(round_to_sf(rate_plot[ii], 3)), linestyle='--', linewidth=2)
    # Add labels, title, and legend
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss Value")
    fig_title = str(data_type) + " fitting loss values from different learning rates and epoch numbers"
    plot_title = "LossPlot" + str(ep_plot[0]) + "to" + str(ep_plot[-1]) + "epochs" + str(round_to_sf(rate_plot[0], 3)) + "to" + str(round_to_sf(rate_plot[-1], 3)) + "rate"
    plt.title(fig_title)
    plt.legend()
    plt.savefig('HW3_NonLinearFit/Plots/' + str(plot_title) + ".png")
    plt.grid(True)


def TrainingLoop(x_generated, y_generated, epochs, learning_rate):
    # Generate data points directly as NumPy arrays (without pandas)
    x_data = x_generated
    y_data = y_generated

    # Reinitialize parameters (n, a, m, b)
    n_fit = np.random.rand()
    a_fit = np.random.rand()
    m_fit = np.random.rand()
    b_fit = np.random.rand()


    # Perform gradient descent for the generated data
    for epoch in range(epochs):
        # Forward pass: compute intermediate and final outputs
        y_int_fit = (m_fit * x_data + b_fit) ** 2
        y_pred_fit = n_fit * np.exp(-a_fit * y_int_fit)

        # Compute gradients
        # Gradients for n and a (output layer)
        grad_n_fit = -2 * np.mean((y_data - y_pred_fit) * np.exp(-a_fit * y_int_fit))
        grad_a_fit = 2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-y_int_fit))

        # Gradients for m and b (inner layer)
        grad_m_fit = 2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit) * x_data))
        grad_b_fit = 2 * np.mean((y_data - y_pred_fit) * n_fit * np.exp(-a_fit * y_int_fit) * (-a_fit) * (2 * (m_fit * x_data + b_fit)))

        # Update parameters
        n_fit -= learning_rate * grad_n_fit
        a_fit -= learning_rate * grad_a_fit
        m_fit -= learning_rate * grad_m_fit
        b_fit -= learning_rate * grad_b_fit

        # Compute loss for monitoring
        loss_fit = compute_loss(x_data, y_data, n_fit, a_fit, m_fit, b_fit)

        # Print loss every 100 epochs
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss_fit:.6f}")

    # Final fitted parameter values
    return n_fit, a_fit, m_fit, b_fit, loss_fit


if __name__ == '__main__':

    # DATA GENERATION ----------------------------------------------------------------------------
    # Seed for reproducibility
    np.random.seed(42)

    # Generate 10 random x values within a range
    x_generated = np.linspace(0, 5, 10)

    # Parameters for the function (can use the previously fitted values or set randomly)
    n_true = 0.06
    a_true = 0.25
    m_true = 0.57
    b_true = 0.11

    # Generate corresponding y values based on the function with added noise
    noise = 0.001 * np.random.normal(0, 0.1, size=x_generated.shape)  # Add Gaussian noise
    y_generated = n_true * np.exp(-a_true * (m_true * x_generated + b_true) ** 2) + noise


    # --------------------------------------------------------------------------------------------

    # Iterating over different epoch and learning rate values to find combination
    epoch_trial = 10             # Number of trials for epoch values
    start_epoch = 0              # Intialized value of epoch number
    epoch_inc = 50              # Scaling of epoch between trial epoch values

    rate_trial = 4              # Number of trials for learning rare values
    start_rate = 0.047              # Initialized value of learning rate
    rate_inc = 0.001           # Scaling of learning rate between trial rate values

    loss_val = np.zeros((epoch_trial, rate_trial))
    # Stores values for losses: 
    #   - Row corresponds to epoch number (increases with row number)
    #   - Column corresponds to learning rate (increases with column number)

    epochs = start_epoch                  # Initializing epoch number
    ep_plot = np.zeros(epoch_trial)
    learning_rate = start_rate            # Initializing learning rate
    rate_plot = np.zeros(rate_trial)

    for ii in range(epoch_trial):
        epochs = epochs + epoch_inc
        ep_plot[ii] = epochs

        for jj in range(rate_trial):
            learning_rate = learning_rate - rate_inc
            #learning_rate = learning_rate * rate_inc
            if ii == 0:
                rate_plot[jj] = learning_rate

            n_fit, a_fit, m_fit, b_fit, loss_fit = TrainingLoop(x_generated, y_generated, epochs, learning_rate)

            #if epochs % 100 == 0:
            if loss_fit < 1e-4:
                NonLinearPlotting(x_generated, n_fit, a_fit, m_fit, b_fit, epochs, learning_rate, loss_fit)

            loss_val[ii][jj] = loss_fit

        learning_rate = start_rate               # Resetting learning rate

    LossPlotting("NonLinear", ep_plot, rate_plot, loss_val)
    print(loss_val)