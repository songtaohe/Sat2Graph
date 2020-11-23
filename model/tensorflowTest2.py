import tensorflow as tf 
import numpy as np 

test_size = 256
tensor_size = 384 

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions())) as sess:
    variables = []
    assign_ops = []
    input_placeholders = []
    arrays = []
    for i in range(test_size):
        variables.append(tf.get_variable("v%d" % i, shape=[3,3,tensor_size, tensor_size], dtype=tf.float32))
        input_placeholders.append(tf.placeholder(tf.float32, shape=[3,3,tensor_size, tensor_size]))
        assign_ops.append(variables[-1].assign(input_placeholders[-1]))
    
        # create some random arrays
        arrays.append(np.random.rand(3,3,tensor_size, tensor_size))

    # main part of the test

    # assign the random array to the tf variables
    sess.run(assign_ops, feed_dict = {i:a for i,a in zip(input_placeholders, arrays)})

    # read the arrays back
    new_arrays = sess.run(variables)

    # check if they match
    succ = True
    for i in range(test_size):
        a = arrays[i]
        b = new_arrays[i]
        
        diff = len(np.where( ((a-b)>0.00001) | ((a-b)<-0.00001) )[0])

        if diff > 0 :
            print(i, "diff=", diff, "max_error", np.amax(a-b), "min_error", np.amin(a-b), "values", b[np.where( ((a-b)>0.00001) | ((a-b)<-0.00001))   ])
            succ = False

           


    if succ:
        print("Test passed")
    else:
        print("Test failed")

    
