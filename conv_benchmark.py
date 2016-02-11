import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn


def get_conv(filter_num, param):
    return nn.Conv2d(filter_num, filter_num, param["kernel_size"],
                     groups=filter_num if param["sep"] else 1,
                     padding=param["padding"], bias=False, dilation=param["dilation"])

image_size = 16
batch_num = 1#32
layer_num = 8#16
filter_nums = [1, 8, 16, 32, 64, 128, 256, 512]
use_gpu = True
# use_gpu = False

params = {
    #"conv11x11": {"kernel_size": 11, "sep": False, "padding": 5, "dilation": (1, 1)},
    #"conv7x7": {"kernel_size": 7, "sep": False, "padding": 3, "dilation": (1, 1)},
    "conv5x5": {"kernel_size": 5, "sep": False, "padding": 2, "dilation": (1, 1)},
    "conv3x3": {"kernel_size": 3, "sep": False, "padding": 1, "dilation": (1, 1)},
    "conv3x3sep": {"kernel_size": 3, "sep": True, "padding": 1, "dilation": (1, 1)},
    "conv1x1": {"kernel_size": 1, "sep": False, "padding": 0, "dilation": (1, 1)},
    "conv1x3": {"kernel_size": (1, 3), "sep": False, "padding": (0, 1), "dilation": (1, 1)},
    "conv3x1": {"kernel_size": (3, 1), "sep": False, "padding": (1, 0), "dilation": (1, 1)},
    "conv3x3dilated": {"kernel_size": (3, 3), "sep": False, "padding": (2, 2), "dilation": (2, 2)},
}

results = {}

for name, param in params.items():
    timings = []

    for filter_num in filter_nums:
        layers = [get_conv(filter_num, param) for _ in range(layer_num)]
        model = nn.Sequential(*layers)
        print(model)
        if use_gpu:
            cudnn.benchmark = True
            cudnn.fastest = True
            input = Variable(torch.randn(batch_num, filter_num, image_size, image_size).cuda())
            model.cuda()            
        else:
            input = Variable(torch.randn(batch_num, filter_num, image_size, image_size))
            
        out = model(input)

        start = time.time()

        for i in range(100):
            out = model(input)

        elapsed_time = time.time() - start
        timings.append(elapsed_time)

    results[name] = timings

for name in params.keys():
    plt.plot(filter_nums, results[name], label=name, marker="o")

plt.xlabel("number of filters")
plt.ylabel("processing time")
plt.legend()
plt.show()
