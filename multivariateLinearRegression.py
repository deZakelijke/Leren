#20151108 H0mework assignment University of Amsterdam
#Second assignment for the course 'leren' of bachelor AI
#Made by Micha de Groot, student number 10434410 and Willemijn Beks, student number 10775110

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

#Read the csv file and convert to right data format.
def useDataFile():
    arr = [[],[],[],[],[]]
    with open('housesRegr.csv', 'rU') as f:
        reader = csv.reader(f, dialect=csv.excel_tab)
        next(reader)
        for row in reader:
            for string in row:
                line = string.split(";")  
                arr[0].append(1)
                arr[1].append(int(line[1]))
                arr[2].append(int(line[2]))
                arr[3].append(int(line[3]))
                arr[4].append(int(line[4]))
    return arr

#Convert the raw data to a matrix
def convertToMatrix(data):
    n = len(data)-1
    m = len(data[0])
    matrix = np.empty([m,n])
    for i in xrange(n):
        for j in xrange(m):
            matrix[j][i]=data[i][j]
    return matrix

#Plot the dataset and the function in two dimensions
def plotData(data, dataType, theta0, theta1):
    plt.plot(data[dataType],data[0],'bo')
    plotPoints = [[]] 
    plotPoints.append([])
    plotPoints[0].append(0)
    plotPoints[0].append(max(data[dataType]))
    plotPoints[1].append(hypothesis(theta0, theta1, 0))
    plotPoints[1].append(hypothesis(theta0, theta1, max(data[dataType])))  
    plt.plot(plotPoints[0],plotPoints[1], 'b-')
    plt.show()

#Calculathe the hypothesis 
def hypothesis(theta, x):
    return np.dot(theta,x)

#Calculate the cost 
def cost(theta, x, y):
    n = len(x)+1
    m = len(x[0])
    cost = 0.0
    for i in xrange(m):
        cost += (hypothesis(theta, x[i]) - y[m])**2
    return cost /(2*m)

#Gradient function for thetaJ
def gradientTheta(theta, x, y, thetaIndex):
    m =  len(x[0])
    gradient = 0.0
    for i in xrange(m):
        gradient += (hypothesis(theta, x[i]) - y[i])*x[i][thetaIndex]
    return gradient/m

#Update theta
def updateTheta(iterations, alpha, theta, x, y):
    tempTheta = theta
    for i in xrange(iterations):
        for j in xrange(len(theta)):   
            tempTheta[j] = theta[j]-alpha * gradientTheta(theta, x, y, j)
        theta = tempTheta
    return theta

def multiple(iterations, alpha, theta):

    print 'number of iterations is set to: ', iterations
    print 'alpha value is set to: ', alpha

    data = useDataFile()
    y = np.array(data[4])
    dataMatrix = convertToMatrix(data[0:5])

    oldCost = cost(theta, dataMatrix, y)
    print 'Old cost is: ', oldCost
    for i in xrange(len(theta)):
        print 'Old theta',i ,' is: ', theta[i]

    newTheta = updateTheta(iterations, alpha, theta, dataMatrix, y)
    newCost = cost(theta, dataMatrix, y)
    print 'New cost is: ', newCost
    for i in xrange(len(theta)):
        print 'New theta',i ,' is: ', theta[i]

    print 'Cost improvement is: ', oldCost - newCost


iterations = 10000
alpha = 0.000000000001
theta = np.array([0.0, 1.0, 1.0, 10.0])

if len(sys.argv)>1:
    iterations = int(sys.argv[1])

if len(sys.argv)>2:
    alpha = float(sys.argv[2])
multiple(iterations, alpha, theta)
