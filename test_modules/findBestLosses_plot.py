# @title plotting the losses related data { vertical-output: true }
import pickle  # to open dict files saved on VM disk

import matplotlib.pyplot as plt


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# load data from Disk
score_vs_losses = load_obj("score_vs_losses.txt")
# print(score_vs_losses)
time_vs_losses = load_obj("time_vs_losses.txt")
# print(time_vs_losses)


plt.figure("Time per epoch for different losses")

for losses in time_vs_losses:
    plt.plot(time_vs_losses[losses], label=str(losses));
plt.title("Time per epoch for different losses")
plt.xlabel("")
plt.ylabel("Time per epoch (s)")
plt.legend()

plt.figure("Score acc for different losses")
for losses in score_vs_losses:
    x = []
    # print(losses)
    # print(score_vs_losses[losses])
    for element in score_vs_losses[losses][2:]:
        if type(element) == list:
            x.append(element[1])
    # print(x)
    plt.plot(x, label=str(losses));
plt.title("Score acc per epoch for different losses")
plt.xlabel("")
plt.ylabel("Score ")
plt.legend()
# save graph to VM disk + Google Drive as PNG
title_file = "Time per epoch for different losses"
plt.savefig(title_file + ".png")

plt.show()
