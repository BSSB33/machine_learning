import json
import sys

#checks length of arguments
if len(sys.argv) < 2:
    print('Imput file required!')
    sys.exit()

with open(sys.argv[1], "r") as input:
    data = json.load(input)

def linkBetween(a, b):
    for link in data['links']:
        if link['points'] == [str(a), str(b)] or link['points'] == [str(b), str(a)]:
            return link

def changeCapacity(route, amount): #list of strings
    for i in range(len(route) - 1):
        #import pdb;pdb.set_trace()
        linkBetween(route[i], route[i+1])['capacity'] += amount
    
#point management
def connectPoints(origin, destination, demand):
    for route in data['possible-circuits']:
        if route[0] == origin and route[-1] == destination: #if link can be established
            found = True
            for i in range(len(route) - 1):
                if linkBetween(route[i], route[i+1])['capacity'] < demand:
                    found = False; break
            if found: #successful branch
                changeCapacity(route, demand * (-1))
                print("igény foglalás: " + str(origin) + "<->" + str(destination) + " st:" + str(currTime) + " - sikeres")
                return route
    print("igény foglalás: " + str(origin) + "<->" + str(destination) + " st:" + str(currTime) + " - sikertelen")
    return []

def endConnection(origin, destination, demand, route):
    if len(route) > 0:
        changeCapacity(route, demand)
        print("igény felszabadítás: " + str(origin) + "<->" + str(destination) + " st:" + str(currTime))

simDurration = int(data["simulation"]["duration"])
for tick in range(1, simDurration):
    for demand in data["simulation"]["demands"]:
        currTime = tick
        if demand in data["simulation"]["demands"]:
            if demand["start-time"] == tick:
                startPoint = demand["end-points"][0] #Letter
                endPoint = demand["end-points"][1] #Letter
                node = connectPoints(startPoint, endPoint, demand["demand"])
                demand["path"] = [] if len(node) < 1 else node
            elif demand["end-time"] == tick:
                endConnection(startPoint, endPoint, demand["demand"], demand["path"])