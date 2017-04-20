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
                time_steps = float(row[x_idx])
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
    parser.add_argument('-f0', '--file0', default='', help='Where the file-logs are stored', dest="file0")
    parser.add_argument('-f1', '--file1', default='', help='Where the file-logs are stored', dest="file1")
    parser.add_argument('-f2', '--file2', default='', help='Where the file-logs are stored', dest="file2")
    parser.add_argument('-f3', '--file3', default='', help='Where the file-logs are stored', dest="file3")
    parser.add_argument('-f4', '--file4', default='', help='Where the file-logs are stored', dest="file4")
    parser.add_argument('-f5', '--file5', default='', help='Where the file-logs are stored', dest="file5")
    parser.add_argument('-f6', '--file6', default='', help='Where the file-logs are stored', dest="file6")
    parser.add_argument('-f7', '--file7', default='', help='Where the file-logs are stored', dest="file7")
    parser.add_argument('-l', '--show_legend', default=False, type=bool, help='Show legend in plot or not',
                        dest="show_legend")
    parser.add_argument('-ws', '--smoothing_window_size', default='100', type=int,
                        help='The window size to use for smoothing', dest="window_size")
    parser.add_argument('-t', '--max_time_steps', default='0', type=int,
                        help='Maximum time steps to show in the plot', dest="max_time_steps")
    args = parser.parse_args()
    
    time_step_idx = 2
    plot_idx = 5

    configs = [{'file_name': args.file0, 'to_print': [time_step_idx, plot_idx], 'color': 'g'}]
    if len(args.file1) > 0:
        configs.append({'file_name': args.file1, 'to_print': [time_step_idx, plot_idx], 'color': 'b'})
    if len(args.file2) > 0:
        configs.append({'file_name': args.file2, 'to_print': [time_step_idx, plot_idx], 'color': 'r'})
    if len(args.file3) > 0:
        configs.append({'file_name': args.file3, 'to_print': [time_step_idx, plot_idx], 'color': 'purple'})
    if len(args.file4) > 0:
        configs.append({'file_name': args.file4, 'to_print': [time_step_idx, plot_idx], 'color': 'y'})
    if len(args.file5) > 0:
        configs.append({'file_name': args.file5, 'to_print': [time_step_idx, plot_idx], 'color': 'm'})
    if len(args.file6) > 0:
        configs.append({'file_name': args.file6, 'to_print': [time_step_idx, plot_idx], 'color': 'k'})
    if len(args.file7) > 0:
        configs.append({'file_name': args.file7, 'to_print': [time_step_idx, plot_idx], 'color': 'c'})

    max_time_steps = None
    if args.max_time_steps > 0:
        max_time_steps = args.max_time_steps

    show_plot(configs, window_size=args.window_size, show_legend=args.show_legend, max_time_steps=max_time_steps)
