import numpy as numpy


def clz_to_prob(clz):
	l = sorted( list( set( clz ) ) ) # クラスを集合にして、リストにし、昇順のソートをかける。クラスの数からなるリスト。
	m = [ l.index( c ) for c in clz ] # クラスの数だけ、リスト中で x と等しい値を持つ最初の要素の位置をゼロから始まる添字で返す。データの数だけある。
	z = np.zeros( (len( clz ), len(l) ) ) # データの数だけ行を持ち、クラスの数だけ列を持つゼロ行列を作成する。

	for i, j in enumerate( m ): # forループの中でリスト（配列）などのイテラブルオブジェクトの要素と同時にインデックス番号（カウント、順番）を取得
		z[i, j] = 1.0 # iにはインデックスの数が、jにはmのの該当する要素（ここでは0か1か）が返される。それに対して1.0を代入する。
	return z, list( map( str, l) ) # ゼロ行列を返す。クラスからなるリストを文字列にしてリスト形式で持つ。



def prob_to_clz( prob, cl):
	i = pred.argmax( axis = 1) # Returns the indices of the maximum values along an axis.axis=1でそのリストの中での最大値を取る。
	return [ cl[z] for z in i ] # 最大をとったリストに合致するものからクラスを選択して返していく。


def get_base_args():
	import argparse # プログラム実行時にコマンドラインで引数を受け取る処理を簡単に実装できる標準ライブラリ
	ps = argparse.ArgumentParser( description = 'ML Test')
	ps.add_argument( '--input', '-i', help='Training file' )
	ps.add_argument( '--separator', '-s', default=',', help='CSV separator' )
	ps.add_argument( '--header', '-e', type=int, default=None, help='CSV header' )
	ps.add_argument( '--indexcol', '-x', type=int, default=None, help='CSV index_col' )
	ps.add_argument( '--regression', '-r', action='store_true', help='Regression' )
	ps.add_argument( '--crossvalidate', '-c', action='store_ture', help='Use Cross Vlidation' )
	return ps


def report_classifier( plf, x, y, clz, cv=True):
	import warnings
	from sklearn.metrics import classification_report, f1_score, accuracy_score
	from sklearn.exceptions import UnderfinedMetricWarning
	from sklearn.model_selection import KFold

	if not cv:
		# モデルとスコアを表示するコード
		plf.fit(x, y)
		print( 'Model:' )
		print( str(plf) )
		z = plf.predict( x )
		z = z.argmax( axis = 1) # axis=1でそのリストの中での最大値を取る。
		y = y.argmax( axis = 1) # axis=1でそのリストの中での最大値を取る。

		with warnings.catch_warnings():
			warnings.simplefilter( 'ignore', category=UnderfinedMetricWarning )
			rp = classification_report( y, z, target_names=clz ) # スコアを取得できる

		print( 'Train Score:' )
		print( rp )

	else:
		# 交差検証のスコアを表示するコード
		kf = kFold( n_splits = 10, random_state = 1, shuffle = True ) # sklearn
		f1 = []
		pr = []
		n = []
		for train_index, test_index in kf.split( x ): # Generate indices to split data into training and test set.
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			plt.fit( x_train, y_train )
			z = plt.predict( x_test )
			z = z.argmax( axis = 1 )
			y_test = y_test.argmax( axis=1 )
			f1.append( f1_score( y_test, z, average='weighted' ) )
			pr.append( accuracy_score( y_test, z ) )
			n.append( len( x_test )/ len( x ) )

		print( 'CV Score:' )
		print( ' F1 Score = %f'%( np.average(f1, weights=n ) ) )
		print( ' Accuracy Score = %f'%( np.average( pr, weights=n ) ) )


def report_regressor( plf, x, y, cv=True ):
	from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
	from sklearn.model_selection import kFold
	if not cv:
		# モデルとスコアを表示するコード
		plt.fit( x, y )
		print( 'Model:' )
		print( str(plf) )
		z = plf.predict( x )
		print( 'Train Score:' )
		rp = r2_score( y, z )
		print( ' R2 Score: %f'%rp )
		rp = explained_variance_score( y, z )
		print( ' Explained Variance Score: %f'%rp )
		rp = mean_absolute_error( y, z )
		print( ' Mean Absolute Error: %f'%rp )
		rp = mean_squared_error( y, z)
		print( ' Mean Squared Error: %f'%rp )

	else:
		# 交差検証のスコアを表示するコード
		kf = kFold( n_splits=10, random_state=1, shuffle=True)
		r2 = []
		ma = []
		n = []
		for train_index, test_index in kf.split( x ):
			x_train, x_test = x[train_index], x[test_index]
			y_train, y_test = y[train_index], y[test_index]
			plt.fit( x_train, y_train)
			z = plf.predict( x_test )
			r2.append( r2_score( y_test, z ) )
			ma.append( mean_squared_error( y_test, z ) )
			n.append( len( x_test )/ len( x ) )
		print( 'CV Score:')
		print( ' R2 Score = %f'%( np.average( r2, weights=n ) ) )
		print( ' Mean Squared Error = %f'%( np.average( ma, weights=n ) ) )
		