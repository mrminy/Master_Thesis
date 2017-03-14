import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


class StatsViewer:

    def __init__(self, configs, args):
        self.configs = configs
        self.col_index = 0
        self.window_size = args.window_size

    def moving_average(self, array):
        window_size = self.window_size
        w = [1.0/window_size]*window_size
        correlate = np.correlate(array, w, 'same')
        for i in reversed(range(window_size)):
            correlate[-i] = np.mean(array[-i:])
        correlate[0] = correlate[1]
        correlate[-1] = correlate[-2]
        return correlate

    def _moving_op(self, array, op, window_size=50):
        array = np.asarray([op(array[i:i + window_size]) for i in range(len(array))])
        array[0] = array[1]
        array[-1] = array[-2]
        return array

    def moving_max(self, array):
        return self._moving_op(array, np.max, self.window_size)

    def moving_min(self, array):
        return self._moving_op(array, np.min, self.window_size)

    def moving_std(self, array):
        return self._moving_op(array, np.std, self.window_size)

    def plot(self):
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
                    ax = plt.subplot(gs[i, 0])
                    plt.yticks(fontsize=14)
                    plt.xticks(fontsize=14)
                    plt.tick_params(axis="both", which="both", bottom="off", top="off",
                                    labelbottom="on", left="off", right="off", labelleft="on")
                    ax.spines["top"].set_visible(False)
                    ax.spines["bottom"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_visible(False)
                    ax.get_xaxis().tick_bottom()
                    ax.get_yaxis().tick_left()
                    ax.set_xlabel(headers[2])
                    if series[name]['type'] == 'value':
                        ax.set_ylabel(headers[plot])
                        asarray = np.asarray(series[name]['data'])
                        moving_std = self.moving_std(asarray)
                        moving_average = self.moving_average(asarray)
                        ax.fill_between(x_series, np.add(moving_average, moving_std), np.add(moving_average, np.multiply(-1.0, moving_std)), facecolor='b', alpha=0.2)
                        ax.plot(x_series, moving_average, color='b')
                        ax.plot(x_series, self.moving_min(asarray), color='r', alpha=0.3)
                        ax.plot(x_series, self.moving_max(asarray), color='g', alpha=0.3)
                    elif series[name]['type'] == 'multiple_values':
                        header = headers[plot].replace('[','').replace(']','')
                        ax.set_title(header)
                        for sub_series, name in zip(series[name]['data'], header.split(',')):
                            ss = np.array(sub_series)
                            ss[ss < -1e20] = 0
                            ss[ss > 1e20] = 0
                            ax.plot(x_series, self.moving_average(ss), label=name)
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
                        mean = self.moving_average(mean)
                        min = self.moving_average(min)
                        max = self.moving_average(max)
                        std = self.moving_average(std)
                        ax.plot(x_series, mean, color='b')
                        ax.plot(x_series, min, color='r')
                        ax.plot(x_series, max, color='g')
                        ax.fill_between(x_series, np.add(mean, std), np.add(mean, np.multiply(-1, std)), facecolor='b', alpha=0.2)
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
        ax = plt.subplot(gs[0, 0])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel(headers[2])
        ax.set_xlabel(x_name)
        ax.set_ylabel('Step')
        asarray = np.asarray(series[y_name]['data'])
        moving_std = self.moving_std(asarray)[::subsample]
        moving_average = self.moving_average(asarray)[::subsample]
        ax.fill_between(x_series, np.add(moving_average, moving_std), np.add(moving_average, np.multiply(-1.0, moving_std)), facecolor='b', alpha=0.2)
        ax.plot(x_series, moving_average, color='b', lw=1.5)

        x_name = 'Relative time'
        y_name = 'Reward'
        x_series = (np.asarray(series[x_name]['data']) / (3.6*60.0*60.0))[::subsample]
        ax = plt.subplot(gs[1, 0])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel(y_name)
        asarray = np.asarray(series[y_name]['data'])
        moving_std = self.moving_std(asarray)[::subsample]
        moving_average = self.moving_average(asarray)[::subsample]
        ax.fill_between(x_series, np.add(moving_average, moving_std), np.add(moving_average, np.multiply(-1.0, moving_std)), facecolor='b', alpha=0.2)
        ax.plot(x_series, moving_average, color='b', lw=1.5)

        for fig in figs:
            fig.show()
        input("Enter to finish")

def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', default='', help='Folder where the logs are stored', dest="folder")
    parser.add_argument('-ws', '--smoothing_window_size', default='50', type=int, help='The window size to use for smoothing', dest="window_size")
    args = parser.parse_args()
    config = [{
        'file_name': args.folder+'env_log_0.txt',
        'to_plot': [5,6]
    }, {
        'file_name': args.folder+'learn_log_0.txt',
        'to_plot': [3,4,5,6]
    }]

    sw = StatsViewer(config, args)
    sw.plot()


