from cProfile import label
import os
from types import new_class

import matplotlib.pyplot as plt


def interpolate(array):
    return None



train_loss = {}
val_loss = {}

with open('data/taco_swin_70epoch.txt', 'r') as f:
    for line in f:
        splitted = line.split('|')

        if len(splitted) != 0:

            if 'train epoch' in splitted[0]:

                train_loss[int(splitted[0].split(' ')[3])] = float(splitted[1].split(' ')[1])

            if 'val epoch' in splitted[0]:

                val_loss[int(splitted[0].split(' ')[3])] = float(splitted[1].split(' ')[1])


        

val_losses = list(val_loss.values())


new_val_losses = [9.645562]

for i, l in enumerate(val_losses):
    new_val_losses.append(l)

    if i != len(val_losses)-1:
        interpolated_value = (l + val_losses[i+1])/2
        new_val_losses.append(interpolated_value)
    

train_losses = list(train_loss.values())

train_losses.pop(len(train_loss)-1)

print(new_val_losses)
xs = [i for i in range(1, 73, 1)]

plt.plot(xs, new_val_losses, label='val')
plt.plot(xs, train_losses, label='train')
plt.legend()

plt.show()

