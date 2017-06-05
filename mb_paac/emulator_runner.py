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

from multiprocessing import Process


class EmulatorRunner(Process):

    def __init__(self, id, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.id = id
        self.emulators = emulators
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:
            instruction = self.queue.get()
            if instruction is None:
                break
            for i, (emulator, action) in enumerate(zip(self.emulators, self.variables[-1])):
                new_s, reward, episode_over = emulator.next(action)
                if episode_over:
                    self.variables[0][i] = emulator.get_initial_state()
                else:
                    self.variables[0][i] = new_s
                self.variables[1][i] = reward
                self.variables[2][i] = episode_over
            count += 1
            self.barrier.put(True)



