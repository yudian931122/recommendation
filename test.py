import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

names = ['userid', 'itemid', 'rating', 'timestamp']
data = pd.read_csv("content_based/data/ratings.dat", sep="::", header=None, names=names)

print(len(data['userid'].unique()))
