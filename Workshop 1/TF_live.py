# TensorFlow Tutorial 2.21
import tensorflow as tf

# Instantiate Session object
sess = tf.Session()

# CREATE ADDER/MULTIPLIER GRAPH
# Initialize placeholder for floats
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# Initiate two operation nodes: adder (shortcut for tf.add) and tripler
# The adder uses the constant placeholder nodes a and b
adder = a + b

# The multiplier uses the output of the adder node
triple = adder * 3

# EVALUATE GRAPH
# Evaluate the adder node using feed dictionaries of different sizes
print(sess.run(adder, {a: 3.5, b: 4}))
print(sess.run(adder, {a: 3.5, b: [2, 4]}))
print(sess.run(adder, {a: [4, 3.5], b: [4, 6]}))

# Evaluate multiplier node for different values of a and b
print(sess.run(triple, {a: 6, b: [5, 2.5]}))
print(sess.run(triple, {a: [6, 14], b: [4, 8]}))


# CREATE GRAPH FOR LINEAR MODEL
# Create variables for unknown weights W and b
# Initial value is usually set randomly. -1 and 1 are the "perfect" weight values for this example
W = tf.Variable([-1], dtype=tf.float32)
b = tf.Variable([1], dtype=tf.float32)

# Initialize constant placeholder for x (known)
x = tf.placeholder(tf.float32)

# Instantiate operation node for linear model
linear = W * x + b

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Evaluate model using least squares cost function
y = tf.placeholder(tf.float32) # Placeholder for ground truth
deltas = tf.square(linear - y) # Squared difference between prediction and ground truth
loss = tf.reduce_sum(deltas) # Sum all of the differences

# EVALUATE LINEAR MODEL
# Run model with feed dictionaries for x. Will output predictions for y.
print(sess.run(linear, {x: [1,2,3,4]}))

# EVALUATE MODEL WITH LOSS
# Run cost function with feed dictionaries for x and corresponding ground truth y
print(sess.run(loss, {x: [1,2,3,4], y: [0, -1, -2, -3]}))