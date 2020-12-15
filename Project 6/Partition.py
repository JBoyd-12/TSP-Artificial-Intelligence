import collections as cs
import math
import random
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Class that determines the fitness and distance of all routes
class Fitness:
    def __init__(self, subset1, subset2):
        self.subset1 = subset1
        self.subset2 = subset2
        self.sum1 = 0
        self.sum2 = 0
        self.fitness = 0

    #function that calculates a fitness score based on how close the sum of the two subsets are
    def individualFitness(self):
        if (self.fitness == 0):
            self.sum1 = sum(self.subset1)
            self.sum2 = sum(self.subset2)
            self.fitness = abs(self.sum1 - self.sum2)
        return self.fitness

#Function that splits the list of numbers into two parts
def partition(numbersList):
    chromosome = []
    for i in range(0, len(numbersList)):
        chromosome.append(0)
    temp = numbersList.copy()
    part1 = []
    part2 = []

    #while loop that randomly selects numbers from the list and adds them to two different list
    while temp:
        number = random.choice(temp)
        part1.append(number)
        temp.remove(number)

        if (len(temp) == 0):
            break
        else:
            number = random.choice(temp)
            part2.append(number)
            temp.remove(number)
    
    for i in range(0, len(numbersList)):
        if numbersList[i] in part2:
            chromosome[i] = 1
    return chromosome

#Function that generates a population of random routes 
def initialPopulation(setOfNumbers, popSize):
    population = []

    #Loop that creates random set of numbers 
    for i in range(0, popSize):
        population.append(partition(setOfNumbers))

    return population

#Finds the fitness and ranks each partition in the population
def rankPartitions(population, setOfNumbers):
    fitnessResults = {}
    part1 = []
    part2 = []

    for i in range(0, len(population)):
        for j in range(0, len(population[i])):
            if (population[i][j] == 0):
                part1.append(setOfNumbers[j])
            if (population[i][j] == 1):
                part2.append(setOfNumbers[j])

        fitnessResults[i] = Fitness(part1, part2).individualFitness()
        part1.clear()
        part2.clear()

    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = False) #returns a sorted list with individual ID's associated with their fitness scores

#Function to select partitions to be used in mating pool from the ranked routes
def selection(rankedPop, eliteSize):
    selectionResults = []
    fitnessTable = pd.DataFrame(np.array(rankedPop), columns = ["Index", "Fitness"])
    fitnessTable['Sum'] = fitnessTable.Fitness.cumsum()
    fitnessTable['Percentage'] = 100*fitnessTable.Sum/fitnessTable.Fitness.sum()
   
    #for loop that adds best partitions into list
    for i in range(0, eliteSize):
        selectionResults.append(rankedPop[i][0])
    #for loop that uses fitness weight to decide if a partition is added to mating pool
    for i in range(0, len(rankedPop) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(rankedPop)):
            if (pick <= fitnessTable.iat[i,3]):
                selectionResults.append(rankedPop[i][0])
                break
    return selectionResults

#Function that extracts chromosomes in the mating pool from the population
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    
    return matingpool

#Function that generates an offspring of two pertitions
def twoPointCrossover(parent1, parent2, point1, point2):
    child = []
 
    for x in range (0, point1):
            child.append(parent1[x])
    for y in range(point1, point2):
        child.append(parent2[y])
    for z in range(point2, len(parent1)):
        child.append(parent1[z])
  
    return child

#Function that creats the offspring population
def newPopulation(matingpool, eliteSize):
    children = []
    point1 = int(len(matingpool)/3)
    point2 = point1 * 2
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    
    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = twoPointCrossover(pool[i], pool[len(matingpool)-i-1], point1, point2)
        children.append(child)
    return children

#Function that has a probability of creating a mutation in a partition
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            if individual[swapped] == 0:
                if individual[swapWith] == 1:
                    bit1 = individual[swapped]
                    bit2 = individual[swapWith]
                    
                    individual[swapped] = bit2
                    individual[swapWith] = bit1
            
            if individual[swapped] == 1:
                if individual[swapWith] == 0:
                    bit1 = individual[swapped]
                    bit2 = individual[swapWith]
                    
                    individual[swapped] = bit2
                    individual[swapWith] = bit1
    return individual

#Function that runs the whole population through the mutation function
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for i in range(0, len(population)):
        mutatedInd = mutate(population[i], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Combines all prior functions to produce next generation of population
def nextGeneration(currentGen, setOfNumbers, eliteSize, mutationRate):
    rankedPopulation = rankPartitions(currentGen, setOfNumbers)
    selectionResults = selection(rankedPopulation, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = newPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

#Loops through creation of generations 
def geneticAlgorithm(setOfNumbers, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(setOfNumbers, popSize)
    print("Initial best sum discrepancy:", rankPartitions(pop, setOfNumbers)[0][1])
    progress = []
   # progress.append(rankPartitions(pop, setOfNumbers)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, setOfNumbers, eliteSize, mutationRate)
        #progress.append(rankPartitions(pop, setOfNumbers)[0][1])

    print("Final best sum discrepancy:",  rankPartitions(pop, setOfNumbers)[0][1])
    bestPartitionIndex = rankPartitions(pop, setOfNumbers)[0][0]
    bestPartition = pop[bestPartitionIndex]

   # plt.plot(progress)
   # plt.ylabel('Cost')
    #plt.xlabel('Generation')
    #plt.show()
    return bestPartition

#Function that fills in final path using Neatest Neighbor algorithm
def createFinalPartition(finalSets, numberSet):
    part1 = []
    part2 = []
    if len(finalSets) == 0:
        while numberSet:
            number = max(numberSet)
            part1.append(number)
            numberSet.remove(number)

            if (len(numberSet) == 0):
                break
            else:
                number = max(numberSet)
                part2.append(number)
                numberSet.remove(number)
    else:
        part1 = finalSets[0].copy()
        part2 = finalSets[1].copy()
    
    sum1 = sum(part1)
    sum2 = sum(part2)
    difference = abs(sum1 - sum2)
    return part1, part2, difference

#Acts as main function
def main():
    lengthOfList = 100 
    maxNumber = 1000
    minNumber = 1
    numberSet = []

    i = 0
    while (i < lengthOfList):
        number = random.randint(minNumber, maxNumber)
        if number not in numberSet:
            numberSet.append(number)
            i = i+1
#-------------------------------------------------------------------------------------------------------------------------------------------------  
    eliteSize = int(lengthOfList/5) 
    mutationRate = 0.12
    generations = 50

    part1 = []
    part2 = []
    finalSets = []
    allSets = []
    fittestIndividuals = 10
    count = 1

    print("Wisdom of the Crowds Algorithm, Solving Partitioning:\n")
    print("Number Set:", numberSet)
    for i in range(0, fittestIndividuals):
        print(count)
        finalPartition = geneticAlgorithm(numberSet, lengthOfList, eliteSize, mutationRate, generations)
        for i in range(0, len(finalPartition)):
            if (finalPartition[i] == 0):
                part1.append(numberSet[i])
            if (finalPartition[i] == 1):
                part2.append(numberSet[i]) 

        print(finalPartition)
        print(part1, part2)
        count = count+1
        
        temp1 = part1.copy()
        temp2 = part2.copy()
        allSets.append(temp1)
        allSets.append(temp2)
        part1.clear()
        part2.clear()
   
      
    for i in range(0, len(allSets)):
        if(i == len(allSets)-1):
            break
        count = 0
        currentSet = allSets[i]

        for j in range(i+1, len(allSets)):
            nextSet = allSets[j]
            if (set(currentSet) == set(nextSet)):
                count = count+1
        if (count >= fittestIndividuals/2):
            if(currentSet in finalSets):
                continue
            else:
                finalSets.append(currentSet)
    
    print("\nCommon Sets:", finalSets)
    finalSolution = createFinalPartition(finalSets, numberSet)
        
    print("The Final Partition:", finalSolution[0], finalSolution[1])
    print("Final Sum Discrepency:", finalSolution[2])

if __name__ == "__main__":
    main()






