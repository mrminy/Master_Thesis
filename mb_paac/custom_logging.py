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
"""


import os
import numpy as np
import time
import json


def load_args(path):
    if path is None:
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_args(args, folder, file_name='args.json'):
    args = vars(args)
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + '/' + file_name, 'w') as f:
        return json.dump(args, f)


class StatsLogger:

    def __init__(self, conf, subfolder='debugging', videos_folder='videos'):
        self.video_count = 0
        self.start_time = time.time()
        self.files = {}
        self.videos_folder = subfolder+'/'+videos_folder
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        for configuration in conf:
            file_name = subfolder+'/'+configuration['file_name']
            if not os.path.isfile(file_name):
                with open(file_name, 'w') as f:
                    f.write(configuration['header'])
            self.files[configuration['name']] = file_name

    def get_stats_for_array(self, array):
        return np.mean(array), np.min(array), np.max(array), np.std(array)

    def _log(self, data, file_name):
        with open(file_name, 'a+') as log_file:
            log_file.write('\n'+data)

    def _to_str(self, arg):
        if hasattr(arg, '__iter__'):
            return ','.join([str(x) for x in arg])
        else:
            return str(arg)

    def _get_timestamped(self, arg):
        current_time = time.time()
        return str(current_time-self.start_time)+'|'+str(current_time)+'|'+arg

    def log(self, name, *args):
        string = self._get_timestamped('|'.join(map(self._to_str, args)))
        file_name=self.files[name]
        self._log(string, file_name)