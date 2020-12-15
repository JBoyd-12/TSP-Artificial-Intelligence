import collections as cs
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creates distance equation
def distance(c1, c2):
    return math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

#Function that plots path of given points and displays them
def plotPath(xCoord, yCoord):
    plt.plot(xCoord, yCoord, 'bo-')
    plt.plot(xCoord[0], yCoord[0], 'rs-')
    plt.axis('scaled'); plt.axis('off')
    plt.show()

#Neatest Neighbor Fucntion
def greedy_tsp(cities, start, aDistances):
    tour = [start]
    unvisited = cities 
    unvisited.remove(start)
    
    while unvisited:
        C = nearestNeighbor(tour[-1], unvisited, aDistances)
        tour.append(C)
        unvisited.remove(C)

    return tour

#Function that calculates closest node of a given node
def nearestNeighbor(C, cities, aDistances):
    distances = []
    for i in range(0,len(cities)):
        distance = aDistances[C][cities[i]]
        distances.append(distance)

    shortest = min(distances)
    index = distances.index(shortest)
    return cities[index]

#Acts as main function
def main():
    #reads in contents from file and creates formatted table
    filename = 'Random200.tsp'
    data = pd.read_table(filename, delimiter = ' ', skiprows = 7, names = ['cityNum','x','y'])     
    coordinates = data[['x', 'y']].reset_index().drop(['index'], axis = 1) #stores the coordinates their own table

    #finds the distances between all cities and stores the values in an array
    numberOfCities = data.count().x
    distances = [[distance(coordinates.iloc[i],coordinates.iloc[j]) 
                  for i in range (numberOfCities)] 
                    for j in range(numberOfCities)] #generates distances between cities
    aDistances = np.asarray(distances) #stores all the distances between every point

    #takes cities and coordinates from table and stores them in individual list
    cities = list(range(0,numberOfCities))
    xCo = []
    yCo = []
    for x in coordinates['x']:
        xCo.append(x)

    for y in coordinates['y']:
        yCo.append(y)
#-------------------------------------------------------------------------------------------------------------------------------------------------
    #uses Greedy Algorithm
    startingCity = 0
    shortestRoute = greedy_tsp(cities, startingCity, aDistances)

    shortestDistance = 0
    for i in range(0,len(shortestRoute)):
        #Sums up the total distance
        shortestDistance += (aDistances[shortestRoute[i-1]][shortestRoute[i]])    

    #creates list that hold x and y coordinates in the order of the shortest route 
    shortX = []
    shortY = []
    for i in shortestRoute:
        shortX.append(xCo[i])
        shortY.append(yCo[i])
    
    shortX.append(shortX[0])
    shortY.append(shortY[0])

    print(filename)
    print("For the Greedy Algorithm:")
    print("The shortest distance is:", shortestDistance)
    print("The shortest route is:", shortestRoute)
    plotPath(shortX, shortY)

if __name__ == "__main__":
    main()



