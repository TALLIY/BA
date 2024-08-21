from datetime import datetime

import numpy as np

import tensorflow as tf

A_dim = np.load(
    "/Users/talli/BA/shared/generated_functions/A_10000_bound(20.0-73.0)_maxmin(0-100).npy"
)

A_dim_tf = tf.convert_to_tensor(A_dim, dtype=tf.float32)
T = np.random.uniform(0, 100, size=101)
T[0] = 20
T[-1] = 73

vec = tf.convert_to_tensor(T)
vec_32 = tf.cast(vec, dtype=tf.float32)
x = tf.Variable(vec_32)

print(A_dim_tf.shape)
print(x.shape)


class MyModule(tf.Module):
    def __init__(self, matrix: tf.Tensor, name=None):
        super().__init__(name=name)
        self.matrix = matrix

    @tf.function
    def mult(self, x: tf.Variable):
        with tf.GradientTape() as tape:
            result = tf.linalg.matvec(self.matrix, x)
            jacobian = tape.jacobian(result, x)
            return jacobian


# You have made a model with a graph!

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = "logs/func/%s" % stamp
writer = tf.summary.create_file_writer(logdir)

# Create a new model to get a fresh trace
# Otherwise the summary will not see the graph.
new_model = MyModule(matrix=A_dim_tf)

# Bracket the function call with
# tf.summary.trace_on() and tf.summary.trace_export().
tf.summary.trace_on(graph=True)
tf.profiler.experimental.start(logdir)
# Call only one tf.function when tracing.
z = print(new_model.mult(x))
with writer.as_default():
    tf.summary.trace_export(name="my_func_trace", step=0, profiler_outdir=logdir)
