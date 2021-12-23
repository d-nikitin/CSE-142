# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:38:16 2020

@author: Dima
"""

"""
PART (a)
I decided to use scikit-learn for this portion 
of the homework.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from math import sqrt
def main():
    """
    PART (b)
    The incercept is 4.360802923546112
    The learned theta vector is [-0.13429383  1.84768377 -0.89658481]
    The RMSE value is 0.18970922173315746
    Below is the code i wroe for this 
    """
    x = np.array( 
       [[3, 9, 2],
        [6, 9, 1],
        [7, 7, 7],
        [8, 6, 4],
        [1, 0, 8]])
    y = np.array([19,19,10,11,-3])
    
    theta = np.array([1,0,8]).reshape(1,-1)
    
    model = LinearRegression().fit(x, y)
    
    ans = model.predict(theta)
    
    print(ans)
    
    inter = model.intercept_
    c = model.coef_
    
    print(inter)
    print(c)
    
    fx = []
    
    for row in x:
        theta = np.array(row).reshape(1,-1)
        fx.append(model.predict(theta))
    
    i = 0
    sigma = 0
    while i < 5:
        sigma+= (((fx[i] - y[i])*(fx[i] - y[i]))/5)
        i+=1
    rms = sqrt(sigma)
    print(rms)
    
    """
    PART (c)
    The label for the instance [3,3,5] is 5.01804869
    Code below
    """
    
    unlabeled = np.array([3,3,5]).reshape(1,-1)
    
    ansC = model.predict(unlabeled)
    
    print(ansC)
    
    """
    PART (D)
    The learned theta vector does not change if the rows of
    x and y are premuted because the data is not interperpreted
    as being order specific so it is the same exact data
    Code below
    """
    
    xP = np.array( 
       [[6, 9, 1],
        [1, 0, 8],
        [7, 7, 7],
        [3, 9, 2],
        [8, 6, 4]])
    yP = np.array([19,-3,10,19,11])
    
    modelP = LinearRegression().fit(xP, yP)
    
    ansP = modelP.coef_
    
    print(ansP)
    
    
    
if __name__ == '__main__':
    main()