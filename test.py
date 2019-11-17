import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

from scipy.stats import bernoulli

print(1 - bernoulli.rvs(0.8, size=(10,10)))