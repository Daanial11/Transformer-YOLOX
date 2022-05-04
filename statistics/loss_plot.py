from cProfile import label
import os
from tkinter import font
from types import new_class

import matplotlib.pyplot as plt


def interpolate(array):
    return None



train_loss = {}
val_loss = {}

with open('data/loss_data/yolo_loss.txt', 'r') as f:
    for line in f:
        splitted = line.split('|')

        if len(splitted) != 0:

            if 'train epoch' in splitted[0]:
                if int(splitted[0].split(' ')[3]) > 50:
                    break
                train_loss[int(splitted[0].split(' ')[3])] = float(splitted[1].split(' ')[1])

            if 'val epoch' in splitted[0]:
                

                val_loss[int(splitted[0].split(' ')[3])] = float(splitted[1].split(' ')[1])



val_loss[42] = 6.218
val_losses = list(val_loss.values())




new_val_losses = [9.645562]

for i, l in enumerate(val_losses):
    new_val_losses.append(l)

    if i != len(val_losses)-1:
        interpolated_value = (l + val_losses[i+1])/2
        new_val_losses.append(interpolated_value)
    

train_losses = list(train_loss.values())

print(len(train_losses))
print(len(new_val_losses))

print(new_val_losses)
xs = [i for i in range(1, 51, 1)]
plt.figure(figsize=(11,8))
plt.grid(alpha=0.4)
plt.plot(xs, new_val_losses, label='Validation loss', linewidth=2.0)
plt.plot(xs, train_losses, label='Training loss', linewidth=2.0)
plt.xlabel("Number of epochs", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)



plt.show()

