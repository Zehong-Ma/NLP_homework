# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Transformer training script."""

import time
from mindspore.train.callback import Callback, TimeMonitor


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))
time_stamp_init = False
time_stamp_first = 0

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss.log", "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                           cb_params.cur_epoch_num,
                                                                           cb_params.cur_step_num,
                                                                           str(cb_params.net_outputs)))
            f.write('\n')
