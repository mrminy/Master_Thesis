"""
Copyright [2017] [Alfredo Clemente]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

--------------------------------------------------------------
Motification
Added better statistical outputs to console. Added extra plot for the dynamics model.
"""


import csv
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class StatsViewer:
    def __init__(self, configs, args):
        self.configs = configs
        self.col_index = 0
        self.window_size = args.window_size
        self.max_timesteps = args.max_timesteps

    @staticmethod
    def moving_average(array, window_size):
        w = [1.0 / window_size] * window_size
        correlate = np.correlate(array, w, 'same')
        for i in reversed(range(window_size)):
            correlate[-i] = np.mean(array[-i:])
        correlate[0] = correlate[1]
        correlate[-1] = correlate[-2]
        return correlate

    @staticmethod
    def moving_op(array, op, window_size=50):
        array = np.asarray([op(array[i:i + window_size]) for i in range(len(array))])
        array[0] = array[1]
        array[-1] = array[-2]
        return array

    @staticmethod
    def moving_max(array, window_size):
        return StatsViewer.moving_op(array, np.max, window_size)

    @staticmethod
    def moving_min(array, window_size):
        return StatsViewer.moving_op(array, np.min, window_size)

    @staticmethod
    def moving_std(array, window_size):
        return StatsViewer.moving_op(array, np.std, window_size)

    def plot(self, extra_figs):
        self.s = []
        figs = []
        for config in self.configs:
            with open(config['file_name'], 'r') as f:
                header = f.readline()
                headers = header.split('|')
                to_plot = config['to_plot']
                to_plot = to_plot + [i for i, h in enumerate(headers) if '*' in h and i not in to_plot]
                fig = plt.figure()
                figs.append(fig)
                gs = gridspec.GridSpec(len(to_plot), 1)
                series = {h: {'type': 'value', 'data': []} for h in headers}
                self.s.append(series)
                lines = f.read().split('\n')
                for plot in to_plot:
                    name = headers[plot]
                    val = lines[0].split('|')[plot]
                    if ',' in val:
                        if ',' in name:
                            series[name]['data'] = [[] for _ in val.split(',')]
                            series[name]['type'] = 'multiple_values'
                        else:
                            series[name]['type'] = 'statistics'
                    else:
                        series[name]['data'] = []
                        series[name]['type'] = 'value'
                for line in lines:
                    cols = line.split('|')
                    for row_nr in range(len(headers)):
                        name = headers[row_nr]
                        current_col = cols[row_nr]
                        if row_nr in to_plot:
                            if series[name]['type'] == 'value':
                                series[name]['data'].append(float(current_col))
                            elif series[name]['type'] == 'multiple_values':
                                for i, e in enumerate(current_col.split(',')):
                                    series[name]['data'][i].append(float(e))
                            elif series[name]['type'] == 'statistics':
                                series[name]['data'].append([float(x) for x in current_col.split(',')])
                        else:
                            series[name]['data'].append(float(current_col))

                x_series = series[headers[2]]['data']
                for i, plot in enumerate(to_plot):
                    name = headers[plot]
                    ax = get_generic_sub_plot(gs[i, 0], headers[2])
                    plt.yticks(fontsize=14)
                    plt.xticks(fontsize=14)
                    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                                    labelbottom="on", left="off", right="off", labelleft="on")
                    ax.get_xaxis().tick_bottom()
                    ax.get_yaxis().tick_left()
                    if series[name]['type'] == 'value':
                        ax.set_ylabel(headers[plot])
                        asarray = np.asarray(series[name]['data'])
                        moving_std = self.moving_std(asarray, self.window_size)
                        moving_average = self.moving_average(asarray, self.window_size)
                        ax.fill_between(x_series, np.add(moving_average, moving_std),
                                        np.add(moving_average, np.multiply(-1.0, moving_std)), facecolor='b', alpha=0.2)
                        ax.plot(x_series, moving_average, color='b')
                        ax.plot(x_series, self.moving_min(asarray, self.window_size), color='r', alpha=0.3)
                        ax.plot(x_series, self.moving_max(asarray, self.window_size), color='g', alpha=0.3)
                    elif series[name]['type'] == 'multiple_values':
                        header = headers[plot].replace('[', '').replace(']', '')
                        ax.set_title(header)
                        for sub_series, name in zip(series[name]['data'], header.split(',')):
                            ss = np.array(sub_series)
                            ss[ss < -1e20] = 0
                            ss[ss > 1e20] = 0
                            ax.plot(x_series, self.moving_average(ss, self.window_size), label=name)
                    elif series[name]['type'] == 'statistics':
                        ax.set_ylabel(headers[plot])
                        sub_series = series[name]['data']
                        mean = []
                        min = []
                        max = []
                        std = []
                        for e in sub_series:
                            mean.append(e[0])
                            min.append(e[1])
                            max.append(e[2])
                            std.append(e[3])
                        mean = self.moving_average(mean, self.window_size)
                        min = self.moving_average(min, self.window_size)
                        max = self.moving_average(max, self.window_size)
                        std = self.moving_average(std, self.window_size)
                        ax.plot(x_series, mean, color='b')
                        ax.plot(x_series, min, color='r')
                        ax.plot(x_series, max, color='g')
                        ax.fill_between(x_series, np.add(mean, std), np.add(mean, np.multiply(-1, std)), facecolor='b',
                                        alpha=0.2)
                    fig.add_axes(ax)

        subsample = self.window_size
        series = self.s[0]
        plt.yticks(fontsize=14)
        plt.xticks(fontsize=14)
        plt.tick_params(axis="both", which="both", bottom="off", top="off",
                        labelbottom="on", left="off", right="off", labelleft="on")
        fig = plt.figure()
        figs.append(fig)
        x_name = 'Global Time Step'
        y_name = 'Reward'
        x_series = series[x_name]['data'][::subsample]
        gs = gridspec.GridSpec(2, 1)
        ax = get_generic_sub_plot(gs[0, 0], x_name, y_name)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        asarray = np.asarray(series[y_name]['data'])
        moving_std = self.moving_std(asarray, self.window_size)[::subsample]
        moving_average = self.moving_average(asarray, self.window_size)[::subsample]
        ax.fill_between(x_series, np.add(moving_average, moving_std),
                        np.add(moving_average, np.multiply(-1.0, moving_std)), facecolor='b', alpha=0.2)
        ax.plot(x_series, moving_average, color='b', lw=1.5)

        x_name = 'Relative time'
        y_name = 'Reward'
        x_series = (np.asarray(series[x_name]['data']) / (3.6 * 60.0 * 60.0))[::subsample]
        ax = get_generic_sub_plot(gs[1, 0], 'Time (hours)', y_name)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        asarray = np.asarray(series[y_name]['data'])
        moving_std = self.moving_std(asarray, self.window_size)[::subsample]
        moving_average = self.moving_average(asarray, self.window_size)[::subsample]
        ax.fill_between(x_series, np.add(moving_average, moving_std),
                        np.add(moving_average, np.multiply(-1.0, moving_std)), facecolor='b', alpha=0.2)
        ax.plot(x_series, moving_average, color='b', lw=1.5)

        if extra_figs is not None:
            for extra_fig in extra_figs:
                if extra_fig is not None:
                    figs.append(extra_fig)

        for fig in figs:
            fig.show()
        input("Enter to finish")


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def generate_special_dynamics_plot(config):
    fig = plt.figure()
    with open(config['file_name'], 'r') as f:
        reader = csv.reader(f.read().split('\n'), delimiter='|')
        header = next(reader)
        if len(header) < 8:
            return None
        x_axis = []
        y_axes = []
        for row in reader:
            x_axis.append(int(row[2]))
            y_axes.append([float(row[7]), float(row[8]), float(row[10]), float(row[11])])
        x_axis = np.array(x_axis)
        y_axes = np.array(y_axes)
        y_axes_moving_avg = []
        for i in range(len(y_axes[0])):
            arr = y_axes[:, i]
            y_axes_moving_avg.append(StatsViewer.moving_average(arr, 50))
        y_axes_moving_avg = np.array(y_axes_moving_avg)

        gs = gridspec.GridSpec(len(y_axes[0]), 1)

        ax = get_generic_sub_plot(gs[0, 0], 'Global times teps', 'Dynamics loss')
        ax.plot(x_axis, y_axes_moving_avg[0])
        fig.add_axes(ax)

        ax = get_generic_sub_plot(gs[1, 0], 'Global time steps', 'Autoencoder loss')
        ax.plot(x_axis, y_axes_moving_avg[1])
        fig.add_axes(ax)

        ax = get_generic_sub_plot(gs[2, 0], 'Global time steps', 'Avg action std')
        ax.plot(x_axis, y_axes_moving_avg[2])
        fig.add_axes(ax)

        ax = get_generic_sub_plot(gs[3, 0], 'Global time steps', 'Avg reward bonus')
        ax.plot(x_axis, y_axes_moving_avg[3])
        fig.add_axes(ax)
    return fig


def get_generic_sub_plot(pos, x_label=None, y_label=None):
    ax = plt.subplot(pos)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    return ax


class StatsPrinter:
    def __init__(self, configs, args):
        self.configs = configs
        self.window_size = args.window_size
        self.max_timesteps = args.max_timesteps

    def print_stats(self):
        out = ''
        for config in self.configs:
            value_idx = config['to_print']
            with open(config['file_name'], 'r') as f:
                reader = csv.reader(f.read().split('\n'), delimiter='|')
                next(reader)
                time_steps = []
                values = []
                for row in reader:
                    t_step = int(row[2])
                    if 0 < self.max_timesteps < t_step:
                        break
                    time_steps.append(t_step)
                    v = []
                    for idx in value_idx:
                        v.append(float(row[idx]))
                    values.append(v)
                time_steps = np.array(time_steps)
                values = np.array(values)

                in_order = True
                for i in range(1, len(time_steps)):
                    if time_steps[i] < time_steps[i - 1]:
                        in_order = False
                        break
                auc_score = None
                if in_order:
                    auc_score = metrics.auc(time_steps, values)
                    auc_score /= time_steps[-1]

                out += str(config) + '\n'
                out += 'Avg: ' + str(np.mean(values, axis=0)) + '\n'
                out += 'AUC: ' + str(auc_score) + '\n'
                out += 'Sum: ' + str(np.sum(values, axis=0)) + '\n'
                out += 'Max: ' + str(np.max(values, axis=0)) + '\n'
                out += 'Time steps: ' + str(time_steps[-1]) + '\n'

        print(out)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='', help='Folder where the logs are stored', dest="folder")
    parser.add_argument('-t', '--max_timesteps', default='0', help='Maximum timesteps for stats printer',
                        dest="max_timesteps", type=int)
    parser.add_argument('-ws', '--smoothing_window_size', default='50', type=int,
                        help='The window size to use for smoothing', dest="window_size")
    args = parser.parse_args()

    configs = [{'file_name': args.folder + 'env_log_0.txt', 'to_print': [5]}]
    printer = StatsPrinter(configs, args)
    printer.print_stats()

    configs = [{
        'file_name': args.folder + 'env_log_0.txt',
        'to_plot': [5, 6]
    }, {
        'file_name': args.folder + 'learn_log_0.txt',
        'to_plot': [3, 4, 5, 6]
    }]

    extra_plot = generate_special_dynamics_plot(configs[-1])

    sw = StatsViewer(configs, args)
    sw.plot(extra_figs=[extra_plot])
