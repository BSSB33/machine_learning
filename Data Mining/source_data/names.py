import random

array = set()
f = open('names.txt', "r")
array = [name for name in f.read().split("\n")]

randomlist = []
for i in range(0, 100):
    n = random.randint(0, len(array))
    randomlist.append(n)

selected_names = []
for i in randomlist:
    selected_names.append(array[i].capitalize())
selected_names.sort()

f = open('selected_names.txt', "w")
for name in selected_names:
    f.write(name + "\n")
f.close()
