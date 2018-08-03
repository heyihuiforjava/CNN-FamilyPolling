import pickle  # to open dict files saved on VM disk
from statistics import *  # for the function mean()

import matplotlib.pyplot as plt


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# load data from Disk
score_vs_batch_size = load_obj("score_vs_batch_size.txt")
# print(score_vs_batch_size)
time_vs_batch_size = load_obj("time_vs_batch_size.txt")
# print(time_vs_batch_size)


plt.figure("Time per epoch for different batch_size")
plt.subplot("121")
for batch_size in time_vs_batch_size:
    plt.plot(time_vs_batch_size[batch_size], label=str(batch_size));
plt.title("Time per epoch for different batch_size")
plt.xlabel("")
plt.ylabel("Time per epoch (s)")
plt.legend()

plt.subplot("122")
x, y = [], []
for batch_size in time_vs_batch_size:
    x.append(int(batch_size))
    y.append(float(mean(time_vs_batch_size[batch_size])))

plt.plot(x, y)
# plt.loglog(x,y, basex=2)
plt.title("Time vs batch size")
plt.xlabel("batch size")
plt.ylabel("Time per epoch (s)")

plt.figure("Score acc for different batch_size")
for batch_size in score_vs_batch_size:
    x = []
    # print(batch_size)
    # print(score_vs_batch_size[batch_size])
    for element in score_vs_batch_size[batch_size][2:]:
        if type(element) == list:
            x.append(element[1])
    # print(x)
    plt.plot(x, label=str(batch_size));
#   leg = ax.legend(['abc'], loc = 'center left', bbox_to_anchor = (1.0, 0.5))
plt.legend()
plt.title("Score acc per epoch for different batch_size")
plt.xlabel("")
plt.ylabel("Score ")

# save graph to VM disk + Google Drive as PNG
title_file = "Time per epoch for different batch_size"
plt.savefig(title_file + ".png")

plt.show()
