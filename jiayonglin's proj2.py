
import sys
import math
import random
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.pyplot as plt
import numpy as np

    
'''
This function is to find the Z with x and y.
'''    
def function_to_optimize(x,y):
    r = math.sqrt(x**2+y**2)
    zpart1 = math.sin(x**2+3*(y**2))/(0.1+r**2)
    zpart2 = (x**2+5*(y**2))*(math.exp(1-r**2)/2)
    result = zpart1+ zpart2
    return result


'''
this function is to find all 8 neighbor from all 8 directions: up, down, left, right, 
upright, upleft, downright, downleft. And store into a set.
'''
def findneighbor(x,y,step):
    allneighbors = set()
    allneighbors.add((x-step,y,function_to_optimize(x-step,y)))
    allneighbors.add((x-step,y-step,function_to_optimize(x-step,y-step)))
    allneighbors.add((x-step,y+step,function_to_optimize(x-step,y+step)))
    allneighbors.add((x,y+step,function_to_optimize(x,y+step)))
    allneighbors.add((x,y-step,function_to_optimize(x,y-step)))
    allneighbors.add((x+step,y,function_to_optimize(x+step,y)))
    allneighbors.add((x+step,y+step,function_to_optimize(x+step,y+step)))
    allneighbors.add((x+step,y-step,function_to_optimize(x+step,y-step)))
    return allneighbors
    
    
    
    

'''
The hill climb function has only one start and stop one the coordinate is in the local min and global min.
'''
def hill_climb(function, step_size, xmin=-2.5, xmax=2.5, ymin =-2.5, ymax = 2.5):
    print("This is the Hill Climb function:")
    #Draw the 3D graph fot the function.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
   
    X = np.arange(-3, 3, 0.1)
    xlen = len(X)
    
    Y = np.arange(-3, 3, 0.1)
    ylen = len(Y)
    
    X, Y = np.meshgrid(X, Y)
    
    R = np.sqrt(X**2 + Y**2)
    
    Z = (np.sin(X**2 + 3*(Y**2))/(0.1 + R**2)) + ((X**2 + 5*(Y**2))*(np.exp(1-R**2)/2))
    colortuple = ('w')
    colors = np.empty(X.shape, dtype=str)
    for y in range(ylen):
        for x in range(xlen):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0, antialiased=False)
   
    #those lists are for draw the route.
    allx = list()
    ally = list()
    allz = list()
    thex = random.uniform(xmin, xmax)
    they = random.uniform(ymin, ymax)
    thebest = function(thex,they)
    
    allx.append(thex)
    ally.append(they)
    allz.append(thebest)
    
    
    checker2 = False
    print("The start coordinate:")
    print("X:",thex)
    print("Y:",they)
    print("Z:",thebest)

    every = set()
    while True:
        checker2 = False
        every = findneighbor(thex, they,step_size)
        for i in every:
        	# if the neighbor's z less than the min we have found, replace it.
            if i[2]< thebest:
                thebest = i[2]
                thex = i[0]
                they = i[1]
               
                allx.append(thex)
                ally.append(they)
                allz.append(thebest)
                
                checker2 = True
        if not checker2:
            break
    print("The final coordinate:")
    print("X",thex)
    print("Y",they)
    print("Z",thebest)
    print()
    print()
    #Draw the route from the start to end
    ax.plot(allx,ally,allz,label='the route')
    
    

    
   
    plt.show()
       
            
            
        
'''
the hill climb random restart has many restart which can find a better min.
'''
def hill_climb_random_restart(function, step_size, num_restarts, xmin=-2.5, xmax=2.5, ymin =-2.5, ymax = 2.5):
    print("This is the Hill Climb Random Restart function:")
    #Draw the 3D graph fot the function.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
   
    X = np.arange(-3, 3, 0.1)
    xlen = len(X)
    
    Y = np.arange(-3, 3, 0.1)
    ylen = len(Y)
    
    X, Y = np.meshgrid(X, Y)
    
    R = np.sqrt(X**2 + Y**2)
    
    Z = (np.sin(X**2 + 3*(Y**2))/(0.1 + R**2)) + ((X**2 + 5*(Y**2))*(np.exp(1-R**2)/2))
    colortuple = ('w')
    colors = np.empty(X.shape, dtype=str)
    for y in range(ylen):
        for x in range(xlen):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0, antialiased=False)
    
    allbest = list()
    
    for j in range(0,num_restarts):
    	#those lists are for draw the route.
        allx = list()
        ally = list()
        allz = list()
        
        thex = random.uniform(xmin, xmax)
        they = random.uniform(ymin, ymax)
        thebest = function(thex,they)
        
        allx.append(thex)
        ally.append(they)
        allz.append(thebest)
    
        checker2 = False

        every = set()
        while True:
            checker2 = False
            every = findneighbor(thex,they,step_size)
            for i in every:
            	# if the neighbor's z less than the min we have found, replace it.
                if i[2]< thebest:
                    thebest = i[2]
                    thex = i[0]
                    they = i[1]
                    
                    allx.append(thex)
                    ally.append(they)
                    allz.append(thebest)
                    
                    checker2 = True
            if not checker2:
                break
        #Draw the route from the start to end
        ax.plot(allx,ally,allz,label='the route')
        thelist = (thex,they,thebest)
        allbest.append(thelist)
    
    thebestbest = allbest[0][2]
    for r in allbest:
        if r[2] < thebestbest:
            thebestx = r[0]
            thebesty = r[1]
            thebestbest = r[2]
    print("The final coordinate:")
    print("X",thebestx)
    print("Y",thebesty)
    print("Z",thebestbest)
    print()
    print()
    
    plt.show()
        



'''
This is simulated_annealing, which is smarter. It can go up with Probability.  
'''    
def simulated_annealing(function, step_size, max_temp, xmin=-2.5, xmax=2.5, ymin =-2.5, ymax = 2.5):
    print("This is simulated_annealing function:")
    #Draw the 3D graph fot the function.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
   
    X = np.arange(-3, 3, 0.1)
    xlen = len(X)
    
    Y = np.arange(-3, 3, 0.1)
    ylen = len(Y)
    
    X, Y = np.meshgrid(X, Y)
    
    R = np.sqrt(X**2 + Y**2)
    
    Z = (np.sin(X**2 + 3*(Y**2))/(0.1 + R**2)) + ((X**2 + 5*(Y**2))*(np.exp(1-R**2)/2))
    colortuple = ('w')
    colors = np.empty(X.shape, dtype=str)
    for y in range(ylen):
        for x in range(xlen):
            colors[x, y] = colortuple[(x + y) % len(colortuple)]
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=colors,
        linewidth=0, antialiased=False)
    #those lists are for draw the route.
    allx = list()
    ally = list()
    allz = list()
    
    thex = random.uniform(xmin, xmax)
    they = random.uniform(ymin, ymax)
    thebest = function(thex,they)
    
    allx.append(thex)
    ally.append(they)
    allz.append(thebest)
    
    print("The start coordinate:")
    print("X:",thex)
    print("Y:",they)
    print("Z:",thebest)
    
    T = max_temp
    for i in range(sys.maxsize):
        #calculate the temperature based on the growing number.
        T = T * (0.99 ** i)
        if T == 0:
            print("The final coordinate:")
            print("X:",thex)
            print("Y:",they)
            print("Z:",thebest) 
            print()
            print()
            break
        every = findneighbor(thex,they,step_size)
        for i in every:
            detla = i[2] - thebest
            #if the neighbor's z smaller than perivous one, replace it. 
            if detla < 0:
                thex = i[0]
                they = i[1]
                thebest = i[2]
                allx.append(thex)
                ally.append(they)
                allz.append(thebest)
            # else, assign the neighbor to the best set with probability, e^(detla/T).    
            else:
                Probability = math.exp(-detla/T)
                #print("the P is :", Probability) 
                if random.random() <= Probability:
                    thex = i[0]
                    they = i[1]
                    thebest = i[2]
                    allx.append(thex)
                    ally.append(they)
                    allz.append(thebest)
    
    #Draw the route from the start to end
    ax.plot(allx,ally,allz,label='the route')
    plt.show()
    
    
def main():
    hill_climb(function_to_optimize,0.01)
    hill_climb_random_restart(function_to_optimize,0.01,10)
    simulated_annealing(function_to_optimize,0.01,100)
    
    

main()



