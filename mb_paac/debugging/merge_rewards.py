"""
Helper script for merging different runs into average data.
"""

import csv
import numpy as np
from matplotlib import pyplot as plt
import custom_logging
import time
import os
from debugging import StatsViewer
from operator import itemgetter


class StatsLogger:
    def __init__(self, conf, subfolder='debugging', videos_folder='videos'):
        self.video_count = 0
        self.start_time = time.time()
        self.files = {}
        self.videos_folder = subfolder + '/' + videos_folder
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        for configuration in conf:
            file_name = subfolder + '/' + configuration['file_name']
            if not os.path.isfile(file_name):
                with open(file_name, 'w') as f:
                    f.write(configuration['header'])
            self.files[configuration['name']] = file_name

    def get_stats_for_array(self, array):
        return np.mean(array), np.min(array), np.max(array), np.std(array)

    def _log(self, data, file_name):
        with open(file_name, 'a+') as log_file:
            log_file.write('\n' + data)

    def _to_str(self, arg):
        if hasattr(arg, '__iter__'):
            return ','.join([str(x) for x in arg])
        else:
            return str(arg)

    def _get_timestamped(self, arg):
        current_time = time.time()
        return str(current_time - self.start_time) + '|' + str(current_time) + '|' + arg

    def log(self, name, *args):
        string = self._get_timestamped('|'.join(map(self._to_str, args)))
        file_name = self.files[name]
        self._log(string, file_name)


def merge_data(configs, max_time_steps, window_size=200, show_plot=False, save_file=None,
               show_legend=False):
    logger_config = [{'name': 'avg', 'file_name': 'avg_log_0.txt',
                      'header': 'Relative time|Absolute time|Global Time Step|empty|Average reward|Average reward moving window'}]
    if save_file is not None:
        stats_logger = custom_logging.StatsLogger(logger_config, subfolder=save_file)

    data_points = []

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
                data_points.append([time_steps, float(row[y_idx])])

            x_axis = np.array(x_axis)
            y_axes = np.array(y_axes)
            y_axes_moving_avg = StatsViewer.moving_average(y_axes, window_size)

            if show_plot:
                plt.plot(x_axis, y_axes_moving_avg, color=config['color'], label=config['file_name'], linewidth=0.2)

    sorted_data_set = sorted(data_points, key=itemgetter(0))
    x_points = [item[0] for item in sorted_data_set]
    y_points = [item[1] for item in sorted_data_set]
    smooth_y_points = StatsViewer.moving_average(y_points, window_size)

    if show_plot:
        plt.plot(x_points, smooth_y_points, color='k', label='average', linewidth=2.0)
        if show_legend:
            plt.legend()
        plt.grid(True)
        plt.show()

    if save_file is not None:
        for i in range(len(sorted_data_set)):
            stats_logger.log('avg', x_points[i], 0, smooth_y_points[i], smooth_y_points[i])


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

    merge_data(configs, args.max_time_steps, window_size=args.window_size, show_plot=args.show_plot,
               save_file=save_path, show_legend=args.show_legend)
