# @title plotting the optimizer related data { vertical-output: true }
import pickle  # to open dict files saved on VM disk

import matplotlib.pyplot as plt


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# load data from Disk
score_vs_optimizer = load_obj("score_vs_optimizer.txt")
# print(score_vs_optimizer)
time_vs_optimizer = load_obj("time_vs_optimizer.txt")
# print(time_vs_optimizer)


plt.figure("Time per epoch for different optimizer")

for optimizer in time_vs_optimizer:
    plt.plot(time_vs_optimizer[optimizer], label=str(optimizer));
plt.title("Time per epoch for different optimizer")
plt.xlabel("")
plt.ylabel("Time per epoch (s)")
plt.legend()

plt.figure("Score acc for different optimizer")
for optimizer in score_vs_optimizer:
    x = []
    # print(optimizer)
    # print(score_vs_optimizer[optimizer])
    for element in score_vs_optimizer[optimizer][2:]:
        if type(element) == list:
            x.append(element[1])
    # print(x)
    plt.plot(x, label=str(optimizer));
plt.title("Score acc per epoch for different optimizer")
plt.xlabel("")
plt.ylabel("Score ")
plt.legend()

# save graph to VM disk + Google Drive as PNG
title_file = "Time per epoch for different optimizer"
plt.savefig(title_file + ".png")

plt.show()
