import csv
import numpy as np
from matplotlib import pyplot as plt
import custom_logging
from debugging import StatsViewer


def merge_data(configs, max_time_steps, window_size=200, samples=50000, show_plot=False, save_file=None, show_legend=False):
    logger_config = [{'name': 'avg', 'file_name': 'avg_log_0.txt',
                      'header': 'Relative time|Absolute time|Global Time Step|empty|Average reward|Average reward moving window'}]
    if save_file is not None:
        stats_logger = custom_logging.StatsLogger(logger_config, subfolder=save_file)

    x_interpolated = np.linspace(0, max_time_steps, num=samples)
    y_points = np.zeros((len(configs), samples))

    for i, config in enumerate(configs):
        x_idx = config['to_print'][0]
        y_idx = config['to_print'][1]

        with open(config['file_name'], 'r') as f:
            reader = csv.reader(f.read().split('\n'), delimiter='|')
            header = next(reader)

            x_axis = []
            y_axes = []
            for row in reader:
                time_steps = int(row[x_idx])
                if max_time_steps is not None and time_steps >= max_time_steps:
                    break
                x_axis.append(time_steps)
                y_axes.append(float(row[y_idx]))

            x_axis = np.array(x_axis)
            y_axes = np.array(y_axes)
            y_axes_moving_avg = StatsViewer.moving_average(y_axes, window_size)
            moving_std = StatsViewer.moving_std(y_axes, window_size)

            if show_plot:
                # plt.fill_between(x_axis, np.add(y_axes_moving_avg, moving_std),
                #                  np.add(y_axes_moving_avg, np.multiply(-1.0, moving_std)), facecolor=config['color'],
                #                  alpha=0.2)
                plt.plot(x_axis, y_axes_moving_avg, color=config['color'], label=config['file_name'], linewidth=0.2)

            for k in range(samples):
                idx = (np.abs(x_axis - x_interpolated[k])).argmin()
                y_points[i][k] = y_axes_moving_avg[idx]

    y_points_avg = np.mean(y_points, axis=0)
    y_points_avg_moving_avg = StatsViewer.moving_average(y_points_avg, window_size)

    if show_plot:
        plt.plot(x_interpolated, y_points_avg_moving_avg, color='k', label='average', linewidth=2.0)
        if show_legend:
            plt.legend()
        plt.show()

    if save_file is not None:
        for s in range(samples):
            stats_logger.log('avg', x_interpolated[s], 0, y_points_avg[s], y_points_avg_moving_avg[s])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f0', '--folder0', default='debugging/', help='Folder where the logs are stored',
                        dest="folder0")
    parser.add_argument('-f1', '--folder1', default='', help='Folder where the logs are stored', dest="folder1")
    parser.add_argument('-f2', '--folder2', default='', help='Folder where the logs are stored', dest="folder2")
    parser.add_argument('-f3', '--folder3', default='', help='Folder where the logs are stored', dest="folder3")
    parser.add_argument('-f4', '--folder4', default='', help='Folder where the logs are stored', dest="folder4")
    parser.add_argument('-f5', '--folder5', default='', help='Folder where the logs are stored', dest="folder5")
    parser.add_argument('-f6', '--folder6', default='', help='Folder where the logs are stored', dest="folder6")
    parser.add_argument('-f7', '--folder7', default='', help='Folder where the logs are stored', dest="folder7")
    parser.add_argument('-sf', '--save_folder', default='', help='Where to save the new log-file', dest="save_folder")
    parser.add_argument('-ws', '--smoothing_window_size', default='100', type=int,
                        help='The window size to use for smoothing', dest="window_size")
    parser.add_argument('-t', '--max_time_steps', default='20000000', type=int,
                        help='Maximum time steps to show in the plot', dest="max_time_steps")
    parser.add_argument('-s', '--samples', default='50000', type=int,
                        help='Number of samples to the average data', dest="samples")
    parser.add_argument('-p', '--show_plot', default=False, type=bool, help='Show plot',
                        dest="show_plot")
    parser.add_argument('-l', '--show_legend', default=False, type=bool, help='Show legend in plot',
                        dest="show_legend")
    args = parser.parse_args()

    time_step_idx = 2
    plot_idx = 5

    configs = [{'file_name': args.folder0 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'g'}]
    if len(args.folder1) > 0:
        configs.append(
            {'file_name': args.folder1 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'b'})
    if len(args.folder2) > 0:
        configs.append(
            {'file_name': args.folder2 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'r'})
    if len(args.folder3) > 0:
        configs.append(
            {'file_name': args.folder3 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'purple'})
    if len(args.folder4) > 0:
        configs.append(
            {'file_name': args.folder4 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'y'})
    if len(args.folder5) > 0:
        configs.append(
            {'file_name': args.folder5 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'm'})
    if len(args.folder6) > 0:
        configs.append(
            {'file_name': args.folder6 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'lime'})
    if len(args.folder7) > 0:
        configs.append(
            {'file_name': args.folder7 + 'env_log_0.txt', 'to_print': [time_step_idx, plot_idx], 'color': 'c'})

    save_path = None
    if len(args.save_folder) > 0:
        save_path = args.save_folder

    merge_data(configs, args.max_time_steps, samples=args.samples, window_size=args.window_size,
               show_plot=args.show_plot, save_file=save_path, show_legend=args.show_legend)
