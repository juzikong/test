import math

a = math.log(100, 10)

print(a)

print("...text output OK...")

file = open('testoutputfile.txt', 'w')
file.write("test\n{}".formate(a))
file.close()
