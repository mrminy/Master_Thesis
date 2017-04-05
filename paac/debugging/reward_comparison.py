import csv
import matplotlib.pyplot as plt
import numpy as np
from debugging import StatsViewer


def show_plot(configs, window_size, show_legend=False, max_time_steps=None):
    for config in configs:
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
            y_axes_moving_avg = []
            y_axes_moving_avg.append(StatsViewer.moving_average(y_axes, window_size))
            y_axes_moving_avg = np.array(y_axes_moving_avg[0])
            moving_std = StatsViewer.moving_std(y_axes, window_size)
            plt.fill_between(x_axis, np.add(y_axes_moving_avg, moving_std),
                             np.add(y_axes_moving_avg, np.multiply(-1.0, moving_std)), facecolor=config['color'],
                             alpha=0.2)
            plt.plot(x_axis, y_axes_moving_avg, color=config['color'], label=config['file_name'])
    if show_legend:
        plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f0', '--folder0', default='', help='Folder where the logs are stored', dest="folder0")
    parser.add_argument('-f1', '--folder1', default='', help='Folder where the logs are stored', dest="folder1")
    parser.add_argument('-f2', '--folder2', default='', help='Folder where the logs are stored', dest="folder2")
    parser.add_argument('-f3', '--folder3', default='', help='Folder where the logs are stored', dest="folder3")
    parser.add_argument('-l', '--show_legend', default=False, type=bool, help='Show legend in plot or not',
                        dest="show_legend")
    parser.add_argument('-ws', '--smoothing_window_size', default='50', type=int,
                        help='The window size to use for smoothing', dest="window_size")
    parser.add_argument('-t', '--max_time_steps', default='0', type=int,
                        help='Maximum time steps to show in the plot', dest="max_time_steps")
    args = parser.parse_args()

    configs = []
    if len(args.folder0) > 0:
        configs.append({'file_name': args.folder0 + 'env_log_0.txt', 'to_print': [2, 5], 'color': 'g'})
    if len(args.folder1) > 0:
        configs.append({'file_name': args.folder1 + 'env_log_0.txt', 'to_print': [2, 5], 'color': 'b'})
    if len(args.folder2) > 0:
        configs.append({'file_name': args.folder2 + 'env_log_0.txt', 'to_print': [2, 5], 'color': 'r'})
    if len(args.folder3) > 0:
        configs.append({'file_name': args.folder3 + 'env_log_0.txt', 'to_print': [2, 5], 'color': 'purple'})

    max_time_steps = None
    if args.max_time_steps > 0:
        max_time_steps = args.max_time_steps

    show_plot(configs, window_size=args.window_size, show_legend=args.show_legend, max_time_steps=max_time_steps)
