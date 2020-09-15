from sklearn import datasets
from sklearn import svm    			
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

X = np.load("/Users/JoshEhrlich/OneDrive - Queen's University/School/University/MSc/PicoscopeAnalysis/OldWork/x.npy")
#Y = np.load("/Users/JoshEhrlich/OneDrive - Queen's University/School/University/Graduate School/Pico Export Data/sample_data/y.npy")

sample1 = X[2]
print(X.shape)
print(Y.shape)
time1 = sample1[1]
volt1 = sample1[0]


sample1 = X[300]
time2 = sample1[1]
volt2 = sample1[0]

# plt.plot(time1, volt1)
# plt.show()

# plt.plot(time2, volt2)
# plt.show()

X_new = np.array([])
Y_new = np.array([])

'''

This is where the manipulation of the waves takes place

'''

#find average "density" of waves
for i in range(1000):
    sample_i = X[i]
    volt_avg_i = np.mean(sample_i[0])
    X_new = np.append(X_new, [volt_avg_i], axis = 0)

#find peaks/crests and then find frequency

avg_frequency = np.array([])
for i in range(1000):
    sample_i = X[i]
    time_i = np.array([])
    for j in range(100):
        if ((sample_i[0][j] > sample_i[0][j-1]) and (sample_i[0][j] > sample_i[0][j+1])):
            time_i = np.append(time_i, [sample_i[1][j]], axis = 0)
    
    length = len(time_i)
    frequency = 100/length
    avg_frequency = np.append(avg_frequency, [frequency])

#amount changed between each point#

pts_difference_avg = np.array([])
for i in range(1000):
	sample_i = X[i]
	pts_difference = np.array([])
	for j in range(99):
		pts_difference = np.append(pts_difference, [abs(sample_i[0][j]-sample_i[0][j+1])], axis = 0)
		
	sample_average = np.mean(pts_difference)
	pts_difference_avg = np.append(pts_difference_avg, [sample_average])

'''

Here I did some brutal building of matrices because I couldnt figure out in-house functions to do that

'''

X_new = np.transpose(X_new)
avg_frequency = np.transpose(avg_frequency)

#Building the classifier array
markers0 = np.full((250),0)
markers1 = np.full((250),1)
markers2 = np.full((250),2)
markers3 = np.full((250),3)

markers = np.array(markers0)
markers = np.append(markers, markers1)
markers = np.append(markers, markers2)
markers = np.append(markers, markers3)
markers = np.transpose(markers)

#Re-formatting arrays to match model requirements
X_new = np.expand_dims(X_new, axis = 0)
X_new = np.transpose(X_new)
avg_frequency = np.expand_dims(avg_frequency, axis = 0)
avg_frequency = np.transpose(avg_frequency)

X_new_and_avg_frequency = np.array([X_new, avg_frequency])
X_new_and_avg_frequency = np.squeeze(X_new_and_avg_frequency)
X_new_and_avg_frequency = np.transpose(X_new_and_avg_frequency)


pts_difference_avg = np.expand_dims(pts_difference_avg, axis = 0)
pts_difference_avg = np.transpose(pts_difference_avg)

X_new_and_pts_diff_avg = np.array([X_new, pts_difference_avg])
X_new_and_pts_diff_avg = np.squeeze(X_new_and_pts_diff_avg)
X_new_and_pts_diff_avg = np.transpose(X_new_and_pts_diff_avg)

'''

This is where I actually performed the SVM. I used the stock code from the website we looked at.

'''

'''
SVM#1 X_new and avg_frequency
'''

C = 1.0
svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_new_and_avg_frequency, markers)
lin_svc = svm.LinearSVC(C=1.0).fit(X_new_and_avg_frequency, markers)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_new_and_avg_frequency, markers)
poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_new_and_avg_frequency, markers)


h = .02  # step size in the mesh
 
# create a mesh to plot in

x_min, x_max = X_new.min() - 1, X_new.max() + 1
y_min, y_max = avg_frequency.min() - 1, avg_frequency.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
# title for the plots
titles = ['SVC with linear kernel',
	   'LinearSVC (linear kernel)',
	    'SVC with RBF kernel',
	    'SVC with polynomial (degree 3) kernel']
 
 
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	 # Plot the decision boundary. For that, we will assign a color to each
	 # point in the mesh [x_min, x_max]x[y_min, y_max].
	 plt.subplot(2, 2, i + 1)
	 plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
	 Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
	 # Put the result into a color plot
	 Z = Z.reshape(xx.shape)
	 plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
	 # Plot also the training points
	 plt.scatter(X_new, avg_frequency, c=markers, cmap=plt.cm.coolwarm)
	 plt.xlabel('X_new')
	 plt.ylabel('avg_frequency')
	 plt.xlim(xx.min(), xx.max())
	 plt.ylim(yy.min(), yy.max())
	 plt.xticks(())
	 plt.yticks(())
	 plt.title(titles[i])
 
plt.show()

'''
SVM#2 X_new and pts_diff_avg
'''


C = 1.0
svc = svm.SVC(kernel = 'linear', C=1.0).fit(X_new_and_pts_diff_avg, markers)
lin_svc = svm.LinearSVC(C=1.0).fit(X_new_and_pts_diff_avg, markers)
rbf_svc = svm.SVC(kernel = 'rbf', gamma = 0.7, C=1.0).fit(X_new_and_pts_diff_avg, markers)
poly_svc = svm.SVC(kernel = 'poly', degree = 3, C = 1.0).fit(X_new_and_pts_diff_avg, markers)


h = .02  # step size in the mesh
 
# create a mesh to plot in

x_min, x_max = X_new.min() - 1, X_new.max() + 1
y_min, y_max = pts_difference_avg.min() - 1, pts_difference_avg.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	                     np.arange(y_min, y_max, h))
# title for the plots
titles = ['SVC with linear kernel',
	   'LinearSVC (linear kernel)',
	    'SVC with RBF kernel',
	    'SVC with polynomial (degree 3) kernel']
 
 
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
	 # Plot the decision boundary. For that, we will assign a color to each
	 # point in the mesh [x_min, x_max]x[y_min, y_max].
	 plt.subplot(2, 2, i + 1)
	 plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
	 Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
 
	 # Put the result into a color plot
	 Z = Z.reshape(xx.shape)
	 plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
	 # Plot also the training points
	 plt.scatter(X_new, pts_difference_avg, c=markers, cmap=plt.cm.coolwarm)
	 plt.xlabel('X_new')
	 plt.ylabel('pts_difference_avg')
	 plt.xlim(xx.min(), xx.max())
	 plt.ylim(yy.min(), yy.max())
	 plt.xticks(())
	 plt.yticks(())
	 plt.title(titles[i])
 
plt.show()

'''

Here is the code where I tried to actually plot a sin function to
the waves however, I could not figure out how to do that

'''

# def test(x, a, b):
#     return a * np.sin(b * x)

# param, param_cov = curve_fit(test, time1, volt1)
# ans = (param[0]*(np.sin(param[1]*time1)))

# #plt.scatter(volt1, time1)
# plt.plot(volt1, ans, '--', color = 'red')
# plt.show()

# print(params)



#how to build those tuples?