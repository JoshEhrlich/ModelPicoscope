from sklearn import datasets
from sklearn import svm    			
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn import metrics
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets

X = np.load("x.npy")
Y = np.load("y.npy")

feat = np.empty([1000,2])

for i in range(0, len(X)):
	data = X[i][0]
	mean = np.mean(data)
	fft_freq = (np.mean(np.fft.fft(data))).real
	feat[i][0] = mean
	feat[i][1] = fft_freq

X_training, X_test, Y_train, Y_test = train_test_split(feat, Y, test_size=0.2, random_state=42)

# plt.xlabel("Mean")
# plt.ylabel("Fourier Transform Frequency")
# plt.scatter(feat[:,0], feat[:,1])
# plt.show()

'''
SVM#1 Mean  vs. FFT
'''

C = 1.0
svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_training, Y_train)
lin_svc = svm.LinearSVC(C=1.0).fit(X_training, Y_train)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_training, Y_train)
poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_training, Y_train)

h = .02  # step size in the mesh
 
# create a mesh to plot in

X_train_min, X_train_max = X_training[:,0].min() - 1, X_training[:,0].max() + 1
Y_train_min, Y_train_max = X_training[:,1].min() - 1, X_training[:,1].max() + 1
X_train, yy = np.meshgrid(np.arange(X_train_min, X_train_max, h),
	                     np.arange(Y_train_min, Y_train_max, h))
# title for the plots
titles = ['SVC with linear kernel',
	   'LinearSVC (linear kernel)',
	    'SVC with RBF kernel',
	    'SVC with polynomial (degree 3) kernel']
 
 
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	 # Plot the decision boundarY_train. For that, we will assign a color to each
	 # point in the mesh [X_train_min, X_train_max]X_train[Y_train_min, Y_train_max].
	 plt.subplot(2, 2, i + 1)
	 plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
	 Z = clf.predict(np.c_[X_train.ravel(), yy.ravel()])
 
	 # Put the result into a color plot
	 Z = Z.reshape(X_train.shape)
	 plt.contourf(X_train, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
	 # Plot also the training points
	 plt.scatter(X_training[:,0], X_training[:,1], c = Y_train, cmap=plt.cm.coolwarm)
	 plt.xlabel('Mean')
	 plt.ylabel('FFT')
	 plt.xlim(X_train.min(), X_train.max())
	 plt.ylim(yy.min(), yy.max())
	 plt.xticks(())
	 plt.yticks(())
	 plt.title(titles[i])
 
# plt.show()

Y_pred = svc.predict(X_test)
print("Accuracy", metrics.accuracy_score(Y_test, Y_pred))

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,4):
	fpr[i], tpr[i], _ = roc_curve(Y_pred, Y_test, pos_label = i)
	roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	#fpr["micro"], tpr["micro"], _ = roc_curve(Y_pred.ravel(), Y_test.ravel(), pos_label = i)
	#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	print(roc_auc)
	plt.figure()
	lw = 2
	plt.plot(fpr[i], tpr[i], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Curve for Objects Fourier Frequency and Mean')
	plt.legend(loc="lower right")
	plt.show()