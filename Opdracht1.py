# Made by Micha de Groot and Danny Dijkzeul
# Python version 2.7 :(

import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

def readData():
    data = []
    with open('housesRegr.csv', 'rU') as csvfile:
        fileReader = csv.reader(csvfile, delimiter=';')
        header = next(fileReader)
        for row in fileReader:
            newRow=[]
            for i in range(1,len(row)):
                newRow.append(int(row[i]))
            data.append(newRow)
    data = np.array(data).T.tolist()
    return data, header[1:]

def plotData(xData, yData, xLabel, yLabel, theta=[]):
    plt.plot(xData,yData, 'bo') 
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)

    # if a theta is give to plot the gradient 
    if theta:
        plotValues = np.arange(0.0, max(xData), 10.0)
        plt.plot(plotValues, theta[0]+theta[1]*plotValues, 'r')
    plt.show()
    return 0 


def gradient(theta, xData, yData, thetaIndex):
    gradientSum = 0
    if thetaIndex == 1:
        for i in range(len(xData)):
            gradientSum += (theta[0] + theta[1]*xData[i]-yData[i])*xData[i]
            return gradientSum / len(xData)
    elif thetaIndex == 0:
         for i in range(len(xData)):
            gradientSum += theta[0] + theta[1]*xData[i]-yData[i]
            return gradientSum / len(xData)

def update(alpha, theta, xData, yData):
    tempTheta = [None]*len(theta)
    for i in range(len(theta)):
        tempTheta[i] = theta[i] - alpha*gradient(theta, xData, yData, i)
    return tempTheta
       

'''
def gradientVector(theta, xVector, yVector, thetaIndex):
    gradientValue = 0
    if thetaIndex == 1:
       gradientValue = theta[0] + theta[1] * xVector  
    elif thetaIndex == 0:
'''
    

def getInputArgs():
    defaultI = 10
    defaultA = 0.0001
    if len(sys.argv) == 0:
        print 'No arguments given. Default to iterations=10 and alpha=0.01'
        print 'oi'
        return defaultI, defaultA
    if len(sys.argv) == 2:
        if not (isinstance(sys.argv[2], int) or isinstance(sys.argv[2], long)):
            print 'Not a valid  #iterations argument. Default to iterations=10 and alpha=0.01'
            return defaultI, defaultA
        print 'Iterations=', sys.argv[2]
        print 'Na argumet for alpha given. Default to alpha=0.01'
        return sys.argv[2], defaultA
    if len(sys.argv) > 2:
        if not (isinstance(sys.argv[2], int) or isinstance(sys.argv[2], long)):
            print sys.argv[2]
            print isinstance(sys.argv[2], int)
            print 'Not a valid #iterations argument. Default to iterations=10 and alpha=0.01'
            return defaultI, defaultA
        if not isinstance(sys.argv[3], float):
            print 'Not a valid alpha argument. Default to iterations=10 and alpha=0.01'
            return defaultI, defaultA
        print 'Iterations=', sys.argv[2]
        print 'Alpha=', sys.argv[3]
        return sys.argv[2], sys.argv[3]


def costFunction(xData, yData, theta):
    costSum = 0
    # assume x and y are aligned in length. i.e. we have correct data
    for i in range(len(xData)):
        costSum += (theta[0]+theta[1]*x[i]-y[i]) **2
    return costSum / (2*len(xData))


def main():
    iterations, alpha = getInputArgs()
    data, labels = readData()
    plotData(data[2],data[-1],labels[2],labels[-1])

    # Theta0=0 and Theta1=1
    theta = [0,1]
    for i in range(iterations):
        theta = update(alpha, theta, data[2], data[-1])
        print theta
    plotData(data[2], data[-1], labels[2], labels[-1], theta)


main()
