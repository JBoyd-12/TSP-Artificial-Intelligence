import collections as cs
import math
import pandas as pd
import numpy as np
import heapq as hq

#Creates distance equation
def distance(c1, c2):
    return math.sqrt((c2[0] - c1[0])**2 + (c2[1] - c1[1])**2)

# BFS function
def bfs_tsp(graph, start, end):
    predecessors = []
    #initializes all the vertexs with infinity distance to start
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0

    heap = [(0, start)]
    while len(heap) > 0:
        currentDistance, currentVertex = hq.heappop(heap)

        if currentDistance > distances[currentVertex]:
            continue

        for neighbor, weight in graph[currentVertex].items():
            dist = currentDistance + weight

            #Only considers new path if it's better than any previous paths
            if dist < distances[neighbor]:
                distances[neighbor] = dist
                predecessors.append(int(neighbor)) 
                hq.heappush(heap, (dist, neighbor))

    return distances

#DFS function
def dfs_tsp(graph, start, end):
    stack = [(start, [start])]
    
    while stack:
        (vertex, path) = stack.pop()
        for nextCity in graph[vertex] - set(path):
            if nextCity == end:
                yield path + [nextCity]
                
            else:
                stack.append((nextCity, path + [nextCity]))

#Acts as main function
def main():
    #reads in contents from file and creates formatted table
    filename = '11PointDFSBFS.tsp'
    data = pd.read_table(filename, delimiter = ' ', skiprows = 7, names = ['cityNum','x','y'])     
    coordinates = data[['x', 'y']].reset_index().drop(['index'], axis = 1) #stores the coordinates in a list

    #finds the distances between all cities and stores the values in an array
    numberOfCities = data.count().x
    distances = [[distance(coordinates.iloc[i],coordinates.iloc[j]) 
                  for i in range (numberOfCities)] 
                    for j in range(numberOfCities)] #generates distances between cities
    aDistances = np.asarray(distances) #stores all the distances between every point
#-------------------------------------------------------------------------------------------------------------------------------------------------

    cities_bfs = {'0': {'1': aDistances[0][1], '2': aDistances[0][2], '3': aDistances[0][3]},
                  '1': {'2': aDistances[1][2]},
                  '2': {'3': aDistances[2][3], '4': aDistances[2][4]},
                  '3': {'4': aDistances[3][4], '5': aDistances[3][5], '6': aDistances[3][6]},
                  '4': {'6': aDistances[4][6], '7': aDistances[4][7]},
                  '5': {'7': aDistances[5][7]},
                  '6': {'8': aDistances[6][8],'9': aDistances[6][9]},
                  '7': {'8': aDistances[7][8],'9': aDistances[7][9],'10': aDistances[7][10]},
                  '8': {'10': aDistances[8][10]},
                  '9': {'10': aDistances[9][10]},
                  '10':{}}

    #uses Breadth First Search algorithm
    start = '0'
    end = '10'
    bfsPath = bfs_tsp(cities_bfs, start, end)

    print("For the BFS Algorithm:")
    print("The distance of the shortest path is:", bfsPath['10'])
    

 #-------------------------------------------------------------------------------------------------------------------------------------------------

    cities_dfs = {'0': set(['1','2','3']),
                  '1': set(['2']),
                  '2': set(['3','4']),
                  '3': set(['4','5','6']),
                  '4': set(['6','7']),
                  '5': set(['7']),
                  '6': set(['8','9']),
                  '7': set(['8','9','10']),
                  '8': set(['10']),
                  '9': set(['10']),
                  '10':set([])}

    #uses Depth First Search algorithm
    startingCity = '0'
    endingCity = '10'
    dfs_list = list(dfs_tsp(cities_dfs, startingCity, endingCity)) #uses DFS to generate all possible paths from first city to last city
    dfs_allPaths = []

    #Loop that turns the values in the dfs_list from str to int
    for path in dfs_list:
        route = []
        for i in path:
            route.append(int(i))
        dfs_allPaths.append(route)

    distanceTraveled = []

    for paths in dfs_allPaths:
        totalDistance = 0
        #Sums up the total distance
        for i in range(0,len(paths)):
            if paths[i] == int(endingCity):
                break
            totalDistance = totalDistance + (aDistances[paths[i]][paths[i+1]])    
        distanceTraveled.append(totalDistance)

    #pulls out the shortest distance and its route
    smallestDistance = min(distanceTraveled)
    shortestRoute = (dfs_allPaths[distanceTraveled.index(smallestDistance)])

    print("\nFor the DFS Algorithm:")
    print("The shortest distance is:", smallestDistance)
    print("The shortest route is:", shortestRoute)

if __name__ == "__main__":
    main()


