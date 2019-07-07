import numpy as np
from collections import Counter

# 標準偏差の計算
def deviation_org(y):
	d = y - y.mean()
	s = d**2

	return np.sqrt(s.mean())

# 標準偏差の計算（短いバージョン）
def deviation(y):
	return y.std()

# ジニ不純物の計算
def gini_org(y):
	i = y.argmax(axis=1)
	clz = set(i)
	c = Counter(i)
	size = y.shape[0]
	score = 0.0
	for val in clz:
		score += (c[val]/size)**2
	return 1.0 - score

# ジニ不純物の計算（高速バージョン）
def gini(y):
	m = y.sum(axis=0)
	size = y.shape[0]
	e = [ (p/size)**2 for p in m]
	return 1.0 - np.sum(e)

# 情報利得の計算
def infgain_org(y):
	i = y.argmax(axis=1)
	clz = set(i)
	c = Counter(i)
	size = y.shape[0]
	score = 0.0
	for val in clz:
		p = c[val]/size
		if p != 0:
			score += p*np.log2(p)
	return -score


# 情報利得の計算（短いバージョン）
def infgain(y):
	m = y.sum(axis=0)
	size = y.shape[0]
	e = [p*np.log2(p/size)/size for p in m if p!= 0.0]
	return -np.sum(e)



