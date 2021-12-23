# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 20:23:44 2020

@author: Dima
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def main():
    # Our dataset and targets
    
    pos_mean = np.array([1,1])
    pos_cov = np.array([[1,0],
                        [0,1]]) 
    neg_mean = np.array([-1,-1])
    neg_cov = np.array([[3,0],
                        [0,3]])
    pos = np.random.multivariate_normal(pos_mean,pos_cov,1000).T
    neg = np.random.multivariate_normal(neg_mean,neg_cov,1000).T
    X = np.concatenate((pos,neg),axis=0)
    X = np.reshape(X,(-2,2))
    
    Y = []
    i = 0 
    while i < 2000:
        if i<1000:
            Y.append(1)
        else:
            Y.append(-1)
        i = i+1 
    # figure number
    fignum = 1
    C = [0.01,0.1,1,10,100,1000]
    # fit the model
    for c in C:
        clf = svm.SVC(kernel='linear', C=c)
        clf.fit(X, Y)
        sw = []
        i = 0
        while i < 2000:
            sw.append(c)
            i = i+1
        # plot the line, the points, and the nearest vectors to the plane
        plt.figure(fignum, figsize=(4, 3))
        plt.clf()
    
        plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80,
                    facecolors='none', zorder=10, edgecolors='k')
        plt.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired,
                    edgecolors='k')
    
        plt.axis('tight')
        x_min = -3
        x_max = 3
        y_min = -3
        y_max = 3
    
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        
        print("Number of Support Vectors for Fig:")
        print(((fignum+1)/2))
        print(len(clf.support_vectors_))
        print("Test Accuracy:")
        print(clf.score(X, Y, sample_weight=sw)*100)
        fignum = fignum +1
        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        plt.figure(fignum, figsize=(4, 3))
        plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                    levels=[-.5, 0, .5])
    
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    
        plt.xticks(())
        plt.yticks(())
        fignum = fignum + 1
    plt.show()
    
if __name__ == '__main__':
    main()