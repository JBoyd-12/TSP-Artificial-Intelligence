import collections as cs
import math
import random
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Creation of city class
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    #Creates distance equation
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        return math.sqrt((xDis**2) + (yDis**2))

    #Displays coordinates
    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

#Class that determines the fitness and distance of all routes
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
    
    #function that will generate the distance of a route
    def routeDistance(self):
        if (self.distance == 0):
            totalDistance = 0
            
            for i in range(0,len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if (i+1 < len(self.route)):
                    toCity = self.route[i+1]
                else:
                    toCity = self.route[0]
                totalDistance += fromCity.distance(toCity)

            self.distance = totalDistance
        return self.distance

    #function that calculates a routes fitness score based on its distance
    def routeFitness(self):
        if (self.fitness == 0):
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

#Function that generates a random route from a list of cities
def generateRoute(cityList):
    return random.sample(cityList, len(cityList))

#Function that generates a population of random routes 
def initialPopulation(popSize, cityList):
    population = []

    #Loops through the generateRoute function
    for i in range(0, popSize):
        population.append(generateRoute(cityList))
    return population

#Finds the fitness and ranks each route in the population
def rankRoutes(population):
    fitnessResults = {}
    
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True) #returns a sorted list with route ID's associated with their fitness score

#Function to select routes to be used in mating pool from the ranked routes
def selection(popRanked, eliteSize):
    selectionResults = []
    fitnessTable = pd.DataFrame(np.array(popRanked), columns = ["Index", "Fitness"])
    fitnessTable['Sum'] = fitnessTable.Fitness.cumsum()
    fitnessTable['Percentage'] = 100*fitnessTable.Sum/fitnessTable.Fitness.sum()

    #for loop that adds best routes into list
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    #for loop that uses fitness weight to decide if a route is added to mating pool
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if (pick <= fitnessTable.iat[i,3]):
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

#Function that extracts routes in the mating pool from the population
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#Function that generates an offspring of two routes
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent2))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

#Function that creats the offspring population
def newPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

#Function that has a probability of creating a mutation in a route
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

#Function that runs the whole population through the mutation function
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for i in range(0, len(population)):
        mutatedInd = mutate(population[i], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Combines all prior functions to produce next generation of population
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = newPopulation(matingpool, eliteSize)
    print(len(children))
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#Loops through creation of generations 
def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance shortest distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final shortest distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

#Creates graph to show improvement of routes over time
def improvementGraph(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.plot(progress)
    plt.ylabel('Cost')
    plt.xlabel('Generation')
    plt.show()

#Function that plots path of given points and displays them
def plotPath(xCoord, yCoord):
    plt.plot(xCoord, yCoord, 'bo-')
    plt.plot(xCoord[0], yCoord[0], 'rs-')
    plt.axis('scaled'); plt.axis('off')
    plt.show()

#Acts as main function
def main():
    #reads in contents from file and creates formatted table
    filename = 'Random100.tsp'
    data = pd.read_table(filename, delimiter = ' ', skiprows = 7, names = ['cityNum','x','y'])     
    coordinates = data[['x', 'y']].reset_index().drop(['index'], axis = 1) #stores the coordinates their own table
    numberOfCities = data.count().x

    #takes cities and coordinates from table and stores them in individual list
    cities = list(range(0,numberOfCities))
    xCo = []
    yCo = []
    for x in coordinates['x']:
        xCo.append(x)

    for y in coordinates['y']:
        yCo.append(y)
#-------------------------------------------------------------------------------------------------------------------------------------------------
    cityList = []

    for i in range(0, numberOfCities):
        cityList.append(City(x=xCo[i], y = yCo[i]))   
    eliteSize = 20 
    mutationRate = 0.001
    generations = 500

    print(filename)
    print("For the Genetic Algorithm:\n")
    finalPath = geneticAlgorithm(cityList, numberOfCities, eliteSize, mutationRate, generations)
    #improvementGraph(cityList, numberOfCities, eliteSize, mutationRate, generations)
   
    #creates list that hold x and y coordinates in the order of the final route 
    shortX = []
    shortY = []
    for i in range(0, len(finalPath)):
        shortX.append(finalPath[i].x)
        shortY.append(finalPath[i].y)
    
    shortX.append(shortX[0])
    shortY.append(shortY[0])

    plotPath(shortX, shortY)

if __name__ == "__main__":
    main()




