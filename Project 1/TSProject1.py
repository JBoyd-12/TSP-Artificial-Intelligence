import itertools as it
import math
import pandas as pd
import numpy as np

def distance(c1, c2):
    return math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

#Acts as main function
def main():
    #reads in contents from file and creates formatted table
    filename = 'Random11.tsp'
    data = pd.read_table(filename, delimiter = ' ', skiprows = 7, names = ['cityNum','x','y'])     
    coordinates = data[['x', 'y']].reset_index().drop(['index'], axis = 1) #stores the coordinates in a list

    #creating the permutations, finding distances, and storing the values in an array
    numberOfCities = data.count().x
    allPaths = list(it.permutations(range(0, numberOfCities))) #generates permutations based on number of cities
    distances = [[distance(coordinates.iloc[i],coordinates.iloc[j]) 
                  for i in range (numberOfCities)] 
                    for j in range(numberOfCities)] #generates distances between cities
    aDistances = np.asarray(distances) #stores all the distances between every point
    
    distanceTraveled = []

    for path in allPaths:
        totalDistance = 0
    
        #Sums up the total distance
        for i in range(0,len(path)):
            totalDistance += (aDistances[path[i-1]][path[i]])
            
        
        distanceTraveled.append(totalDistance)
    
    #pulls out the shortest distance and its route
    smallestDistance = min(distanceTraveled)
    shortestRoute = (allPaths[distanceTraveled.index(smallestDistance)])

    print(filename)
    print("The shortest distance is:", smallestDistance)
    print("The shortest route is:", shortestRoute)
    

if __name__ == "__main__":
    main()

