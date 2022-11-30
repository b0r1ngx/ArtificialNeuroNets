# import tensorflow as tf
#
#
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import random as r

# l = [1, 5, 6]
# for i in range(len(l)):
#     l[i] = l[i] + 1
#
# print(l)
#
# print(type(range(len(l))))
# print(1 / 4)
# print(1 / 8)
from numpy import array


def invert_with_k_chance(i, k):
    if r.random() <= k:
        if i == 0:
            return 1
        else:
            return 0
    return i


def invert_with_k_chance(i, k, number_of_class=2):
    classes = [i for i in range(number_of_class)]
    if r.random() <= k:
        if i == 0:
            classes.remove(0)
            return r.choice(classes)
        elif i == 1:
            classes.remove(1)
            return r.choice(classes)
        elif i == 2:
            classes.remove(2)
            return r.choice(classes)
        elif i == 3:
            classes.remove(3)
            return r.choice(classes)
        elif i == 4:
            classes.remove(4)
            return r.choice(classes)
        elif i == 5:
            classes.remove(5)
            return r.choice(classes)
        elif i == 6:
            classes.remove(6)
            return r.choice(classes)
        elif i == 7:
            classes.remove(7)
            return r.choice(classes)
        elif i == 8:
            classes.remove(8)
            return r.choice(classes)
        elif i == 9:
            classes.remove(9)
            return r.choice(classes)
    return i


rand = r.random()

classes = [i for i in range(0)]
print(classes)
classes.remove(0)
print(classes)

r = r.choice(classes)
print(r)

b = array([0])
print(b)
print(int(b))