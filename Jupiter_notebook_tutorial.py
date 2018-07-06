# Imports
import numpy
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline
## use `%matplotlib notebook` for interactive figures
# plt.style.use('ggplot')
# import sklearn
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction