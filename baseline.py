import re
import numpy as np
import pandas as pd
import support
from sklearn.model_selection import KFold, cross_validate
from sklearn.svm import SVC, SVR
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklean.neaural_network import MLPClassifier, MLPRegressor

if __name__ == '__main__':
	#
