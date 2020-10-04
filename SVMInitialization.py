import os
from sklearn import datasets
from sklearn import svm    			
import numpy as np
import scipy as sy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn import metrics
import numpy.fft as fftt
from scipy.fftpack import fft
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from scipy.signal import butter,filtfilt
import chart_studio.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
import glob
import numpy as np
import pandas as pd
import scipy

from scipy import signal


dir_path = "/Users/JoshEhrlich/OneDrive - Queen's University/School/University/MSc/PicoscopeAnalysis/ModelPicoscope/Data/Data/"
#os.chdir(os.path.join(dir_path, 'data'))


os.chdir(dir_path)
listOfFiles = glob.glob("*.txt")
channelAAirCut = np.zeros((2504,5))
channelBAirCut = np.zeros((2504,5))
channelATissueCut = np.zeros((2504,5))
channelBTissueCut = np.zeros((2504,5))
channelAAirCoag = np.zeros((2504,5))
channelBAirCoag = np.zeros((2504,5))
channelATissueCoag = np.zeros((2504,5))
channelBTissueCoag = np.zeros((2504,5))

cu_aCount = 0
cu_tCount = 0
co_aCount = 0
co_tCount = 0
for file in listOfFiles:
    if "cut_air" in file:
        cut_air = np.loadtxt(file)
        channelA = cut_air[:,1]
        channelB = cut_air[:,2]
        channelAAirCut[:,cu_aCount] = channelA
        channelBAirCut[:,cu_aCount] = channelB
        cu_aCount += 1
    elif "cut_tissue" in file:
        cut_tissue = np.loadtxt(file)
        channelA = cut_tissue[:,1]
        channelB = cut_tissue[:,2]
        channelATissueCut[:,cu_tCount] = channelA
        channelBTissueCut[:,cu_tCount] = channelB
        cu_tCount += 1
    elif "coag_air" in file:
        coag_air = np.loadtxt(file)
        channelA = coag_air[:,1]
        channelB = coag_air[:,2]
        channelAAirCoag[:,co_aCount] = channelA
        channelBAirCoag[:,co_aCount] = channelB
        co_aCount += 1
    else:
        coag_tissue = np.loadtxt(file)
        channelA = coag_tissue[:,1]
        channelB = coag_tissue[:,2]
        channelATissueCoag[:,co_tCount] = channelA
        channelBTissueCoag[:,co_tCount] = channelB
        co_tCount += 1

#Visualization:

plt.plot(timeAirCut, channelBAirCut)
plt.title("Air_Cut")
#plt.show()

plt.plot(timeTissueCut, channelBTissueCut)
plt.title("Tissue_Cut")
#plt.show()

plt.plot(timeAirCoag, channelBAirCoag)
plt.title("Air_Coag")
#plt.show()

plt.plot(timeTissueCoag, channelBTissueCoag)
plt.title("Tissue_Coag")
#plt.show()

def mean(channel):

    mean = np.mean(channel)

    return mean

def fft(channel):

    frequency = np.mean((np.fft.fft(channel))).real

    return frequency

def testFreq(channel):

    testFreq = sy.fft(channel).real

    return testFreq

def minumum(channel):

    minimum = np.minimum(channel)

    return minimum

def maximum(channel):

    maximum = np.amax(channel)

    return maximum

def absSum(channel):

    absSum = np.sum(np.absolute(channel))

    return absSum

def absMean(channel):

    absMean = np.mean(np.absolute(channel))

    return absMean
    
def stdev(channel):

    stdev = np.std(channeL)
    
    return stdev
    
def absStdev(channel):

    absStdev = np.stdev(np.absolute(channel))
    
    return absStdev

def lmrSum(channelA, channelB):
    
    lmrSum = absSum(channelA) - absSum(channelB)
    
    return lmrSum

def lmrMean(channelA, channelB):
    
    lmrMean = absMean(channelA) - absSum(channelB)
    
    return lmrMean


#idk what this testing was. saw it online.
spectrumA = fftt.fft(channelBAirCut)
freqA = fftt.fftfreq(len(spectrumA))
threasholdA = 0.5 * max(abs(spectrumA))
maskA = abs(spectrumA) > threasholdA
peaksA = freqA[maskA]

spectrumT = fftt.fft(channelBTissueCut)
freqT = fftt.fftfreq(len(spectrumT))
threasholdT = 0.5 * max(abs(spectrumT))
maskT = abs(spectrumT) > threasholdT
peaksT = freqT[maskT]
plt.plot(freqT, abs(spectrumT))
#plt.show()

#print("A", peaksA, "T", peaksT)

lmrMeanA = lmrMean(channelAAirCut, channelBAirCut)
lmrMeanT = lmrMean(channelATissueCut, channelBTissueCut)

print("air:", lmrMeanA,"tissue:", lmrMeanT)

feat = np.empty([200,2])

"""
training:

extract all files of same type cuA, cuT, coA, coT (how many do I need?)
    a. get a list of file names in given folder
    b. save type (0/1/2/3 = cuA/cuT/coA/coT)
    c. run analysis and get 1 number for each function on every file. Thus where x = number of functions, we have x * files of numbers

"""
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
Y = Y.squeeze()

X_training, X_test, Y_train, Y_test = train_test_split(feat, Y, test_size=0.2, random_state=42)

'''
#SVM#1 Mean  vs. FFT
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
'''
