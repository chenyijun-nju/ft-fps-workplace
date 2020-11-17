import tensorflow  as ts 
ts.compat.v1.disable_eager_execution()
a = ts.constant(5)
b = ts.constant(2)
c = ts.constant(3)
d = ts.multiply(a,b)
e = ts.add(c,b)
f = ts.subtract(d,e)
with ts.compat.v1.Session() as sess:
    fetches = [a,b,c,d,e,f]
    outs=sess.run(fetches)
print("outs = {}".format(outs))
print(type(outs[0]))
