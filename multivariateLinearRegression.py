#20151101 H0mework assignment University of Amsterdam
#First assignment for the course 'leren' of bachelor AI
#Made by Micha de Groot, student number 10434410

import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

#Read the csv file and convert to right data format.
def useDataFile():
    arr = [[]]
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
    for i in xrange(data):
        for j in xrange(data[0])
            matrix[i][j]=data[j][i]
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
    return theta.dot(x)

#Calculate the cost 
def cost(theta, data):
    m = len(data[0])
    price = 0.0
    for i in xrange(m):
        price += (hypothesis(theta0, theta1, data[dataType][i]) - data[0][i])**2
    return price /(2*m)

#Gradient function for theta0
def gradientTheta0(theta0, theta1, data, dataType):
    m =  len(data[0])
    gradient = 0.0
    for i in xrange(m):
        gradient += hypothesis(theta0, theta1, data[dataType][i]) - data[0][i]
    return gradient/m

#Update theta0
def updateTheta0(alpha, theta0, theta1, data, dataType):
     return theta0-alpha * gradientTheta0(theta0, theta1, data, dataType)

#Iterate to update theta0 and theta1
def updateParameters(iterations, alpha, theta0, theta1, data, dataType):
    tempTheta0 = theta0
    for i in xrange(iterations):
        theta0 = updateTheta0(alpha, theta0, theta1, data, dataType)
        theta1 = updateTheta1(alpha, tempTheta0, theta1, data, dataType)
        tempTheta0 = theta0
    return [theta0, theta1]

#main program
iterations = 10
alpha = 0.001
theta = np.array([0.0, 1.0, 1.0, 1.0])

if len(sys.argv)>1:
    iterations = int(sys.argv[1])

if len(sys.argv)>2:
    alpha = float(sys.argv[2])

print 'number of iterations is set to: ', iterations
print 'alpha value is set to: ', alpha

data = useDataFile()
y = np.array(data[4])
dataMatrix = convertToMatrix(data[0:3])

print 'Old cost is: ', cost(theta0, theta1, data, dataType)
newParam = updateParameters(iterations, alpha, theta0, theta1, data, dataType)
theta0 = newParam[0]
theta1 = newParam[1]
print 'New theta0 is: ', theta0
print 'New theta1 is: ', theta1
print 'New cost is: ', cost(theta0, theta1, data, dataType)
plotData(data,dataType, theta0, theta1)
