from uncertainty_experiments.network_architectures import generate_samples, Bootstrap


def main(n_samples=200):
    X, Y = generate_samples(n_samples)  # Training samples

    # Parameters
    learning_rate = 0.0001
    training_epochs = 2000
    batch_size = 10
    display_step = 1
    keep_prob = 0.95

    model = Bootstrap(1, 1, learning_rate, X, Y, nb_heads=4)
    loss_history = model.train(training_epochs, batch_size=batch_size, display_step=display_step, keep_prob=keep_prob)

    model.test(plot=True, use_mc_dropout=True, n_stochastic_forward_passes=100, keep_prob=keep_prob)

    # Plot cost history
    # plot_x = np.arange(len(cost_mc_dropout))
    # plt.plot(plot_x, cost_mc_dropout, plot_x, cost_bootstrap)
    # plt.show()


if __name__ == '__main__':
    main(n_samples=2000)
