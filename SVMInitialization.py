import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("/Users/JoshEhrlich/OneDrive - Queen's University/School/University/MSc/PicoscopeAnalysis/Data/data/valley_cut/20us/")

air = np.loadtxt('20us_air_01.txt')
tissue = np.loadtxt('20us_tissue_01.txt')

timeAir = air[:,0]
channelAAir = air[:,1]
channelBAir = air[:,2]

timeTissue = tissue[:,0]
channelATissue = tissue[:,1]
channelBTissue = tissue[:,2]

plt.plot(timeAir, channelAAir)
plt.show()

plt.plot(timeAir, channelBAir)
plt.show()

plt.plot(timeTissue, channelATissue)
plt.show()

plt.plot(timeTissue, channelBTissue)
plt.show()