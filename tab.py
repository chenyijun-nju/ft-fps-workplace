import tensorflow as tf 
tf.compat.v1.disable_eager_execution()
A = tf.constant([[1,2,3],[4,5,6]])
print(A.get_shape())
x = tf.constant([1,0,1])
print(x.get_shape())

x = tf.expand_dims(x,1)
print(x.get_shape())

b = tf.matmul(A,x)

sess = tf.compat.v1.InteractiveSession()
print('matul result:\n {}'.format(b.eval()))
print('x:\n {}'.format(x.eval()))
sess.close()