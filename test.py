from time import sleep
import numpy as np


def delete_line():
    print("\033[1A", end="")


print("Hi !")
print("How are you ?")
sleep(1)

LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

print(LINE_UP, end=LINE_CLEAR)

print("New Text")

print(np.floor(9.99))
