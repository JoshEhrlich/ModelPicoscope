import os
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

dir_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(dir_path, 'data'))

airCut = np.loadtxt('20us_cut_air_01.txt')
tissueCut = np.loadtxt('20us_cut_tissue_01.txt')

airCoag = np.loadtxt('20us_coag_air_01.txt')
tissueCoag = np.loadtxt('20us_coag_tissue_01.txt')

#Break up array

timeAirCut = airCut[:,0]
channelAAirCut = airCut[:,1]
channelBAirCut = airCut[:,2]

timeTissueCut = tissueCut[:,0]
channelATissueCut = tissueCut[:,1]
channelBTissueCut = tissueCut[:,2]

timeAirCoag = airCoag[:,0]
channelAAirCoag = airCoag[:,1]
channelBAirCoag = airCoag[:,2]

timeTissueCoag = tissueCoag[:,0]
channelATissueCoag = tissueCoag[:,1]
channelBTissueCoag = tissueCoag[:,2]

#Visualization:

print("")
plt.plot(timeAirCut, channelBAirCut)
plt.title("Air_Cut")
plt.show()

plt.plot(timeTissueCut, channelBTissueCut)
plt.title("Tissue_Cut")
plt.show()

plt.plot(timeAirCoag, channelBAirCoag)
plt.title("Air_Coag")
plt.show()

plt.plot(timeTissueCoag, channelBTissueCoag)
plt.title("Tissue_Coag")
plt.show()


feat = np.empty([200,2])

for i in range(0, 50):
    #How do I make this cleaner?
    data = channelBAirCut[i*50:50+i*50]
    mean = np.mean(data)
    fft_freq = (np.mean(np.fft.fft(data))).real
    feat[i][0] = mean
    feat[i][1] = fft_freq
    data = channelBTissueCut[i*50:50+i*50]
    mean = np.mean(data)
    fft_freq = (np.mean(np.fft.fft(data))).real
    feat[i+50][0] = mean
    feat[i+50][1] = fft_freq
    data = channelBAirCoag[i*50:50+i*50]
    mean = np.mean(data)
    fft_freq = (np.mean(np.fft.fft(data))).real
    feat[i+100][0] = mean
    feat[i+100][1] = fft_freq
    data = channelBTissueCoag[i*50:50+i*50]
    mean = np.mean(data)
    fft_freq = (np.mean(np.fft.fft(data))).real
    feat[i+150][0] = mean
    feat[i+150][1] = fft_freq

#Build classifier array

zero = np.full((50,1),0)
one = np.full((50,1),1)
two = np.full((50,1),2)
three = np.full((50,1),3)

Y = np.concatenate((zero,one,two,three))

X_training, X_test, Y_train, Y_test = train_test_split(feat, Y, test_size=0.2, random_state=42)

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