import math
import gc
import sys
import warnings
import os
import errno
import numpy as np
import time
from scipy.integrate import quad
import importlib

a = math.log(100, 10)

print(a)

print("...text output OK...")

file = open('testoutputfile.txt', 'w')
file.write("out put file generated. \n\n===\n\n")
file.close()
