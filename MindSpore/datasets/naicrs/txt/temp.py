import os

f = open("valC.txt", "r+")
lines = [line[:-1] for line in f.readlines()]
basenames = [os.path.splitext(line)[0] for line in lines]
f.close()
f = open("valC.txt", "w+")
for basename in basenames:
    f.write("images/"+basename+".tif masks/"+basename+".png\n")
f.close()