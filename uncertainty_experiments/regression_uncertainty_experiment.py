from uncertainty_experiments.network_architectures import MCDropout, generate_samples, Bootstrap
import matplotlib.pyplot as plt
import numpy as np

def main(n_samples=100):
    X, Y = generate_samples(n_samples)  # Training samples


    # Parameters
    learning_rate = 0.0001
    training_epochs = 2000
    batch_size = 1
    display_step = 200
    keep_prob = 0.95

    mc_dropout_model = Bootstrap(1, 1, learning_rate, X, Y, nb_heads=4)
    mc_dropout_model.train(training_epochs, batch_size=batch_size, display_step=display_step, keep_prob=keep_prob)
    mc_dropout_model.test(use_mc_dropout=True, n_stochastic_forward_passes=100, keep_prob=keep_prob)


    # Test MC dropout
    # mc_dropout_model = MCDropout(1, 1, learning_rate, X, Y)
    # cost_mc_dropout = mc_dropout_model.train(training_epochs, batch_size=batch_size, display_step=display_step, keep_prob=keep_prob)
    # mc_dropout_model.test(n_stochastic_forward_passes=1, keep_prob=1.)  # Regular feed-forward

    # MC dropout
    # mc_dropout_model.test(n_stochastic_forward_passes=10, keep_prob=keep_prob)
    # mc_dropout_model.test(n_stochastic_forward_passes=100, keep_prob=keep_prob)
    # mc_dropout_model.test(n_stochastic_forward_passes=250, keep_prob=keep_prob)
    # mc_dropout_model.close_session()

    # bootstrap_model = Bootstrap(1, 1, learning_rate, X, Y, nb_heads=50)
    # cost_bootstrap = bootstrap_model.train(training_epochs, batch_size=batch_size, display_step=display_step, keep_prob=keep_prob)
    # bootstrap_model.test(use_mc_dropout=False)  # Only bootstrap

    # Bootstrap & MC dropout
    # bootstrap_model.test(use_mc_dropout=True, n_stochastic_forward_passes=10, keep_prob=keep_prob)
    # bootstrap_model.test(use_mc_dropout=True, n_stochastic_forward_passes=100, keep_prob=keep_prob)
    # bootstrap_model.test(use_mc_dropout=True, n_stochastic_forward_passes=250, keep_prob=keep_prob)
    # bootstrap_model.close_session()

    # Plot cost history
    # plot_x = np.arange(len(cost_mc_dropout))
    # plt.plot(plot_x, cost_mc_dropout, plot_x, cost_bootstrap)
    # plt.show()


if __name__ == '__main__':
    main(n_samples=2000)
