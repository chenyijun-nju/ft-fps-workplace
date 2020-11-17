import math
def quadratic(a,b,c):
    x1=int((-b+math.sqrt(b*b-4*a*c))/(2*a))
    x2=int((-b-math.sqrt(b*b-4*a*c))/(2*a))
    return x1,x2
print("value\n")
a=int(input('a\t'))
b=int(input('\nb\t'))
c=int(input('\nc\t'))
x1,x2=quadratic(a,b,c)
print('solution=',x1,x2)