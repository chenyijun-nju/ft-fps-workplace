def maxmin(L):
    max=L[0]
    min=max
    for value in L:
        if value>max:
            max=value
        if value<min:
            min=value
    return(max,min)        
L=(1,2,3,4,5,7,8,9,3)
print(maxmin(L)[0],maxmin(L)[1])
