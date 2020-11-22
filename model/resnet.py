import tensorflow as tf
import tensorflow.contrib as tf_contrib


# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)

#weight_regularizer = tf_contrib.layers.l2_regularizer(0.000001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias, padding=padding)

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        x = relu(x)


        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        x = relu(x)
        x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')



        return x + x_init

def bottle_resblock(x_init, channels, is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        shortcut = relu(x)

        x = conv(shortcut, channels, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        x = relu(x)
        x = conv(x, channels*4, kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    #return x 
    #print("resnet batch norm hit!")
    return tf_contrib.layers.batch_norm(x,
                                       decay=0.99, epsilon=1e-05,
                                       center=True, scale=True, updates_collections=None,
                                       is_training=is_training, scope=scope, fused = False)

##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit))
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return loss, accuracy


# def resnet(x, is_training=True, reuse=False, res_n = 18, feature_size = 126):
#     with tf.variable_scope("resnet", reuse=reuse):

#         if res_n < 50 :
#             residual_block = resblock
#         else :
#             residual_block = bottle_resblock

#         residual_list = get_residual_layer(res_n)

#         ch = 16 # paper is 64
#         x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

#         for i in range(residual_list[0]) :
#             x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

#         ########################################################################################################

#         x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

#         for i in range(1, residual_list[1]) :
#             x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

#         ########################################################################################################

#         x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

#         for i in range(1, residual_list[2]) :
#             x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

#         ########################################################################################################

#         x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

#         for i in range(1, residual_list[3]) :
#             x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

#         ########################################################################################################


#         x = batch_norm(x, is_training, scope='batch_norm')
#         x = relu(x)

#         x = global_avg_pooling(x)
#         x = fully_conneted(x, units=feature_size, scope='logit')

#         return x


def resnet(x, is_training=True, reuse=False, res_n = 18, feature_size = 126, ch = 64):
    with tf.variable_scope("resnet", reuse=reuse):

        if res_n < 50 :
            residual_block = resblock
        else :
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        #ch = 8 # paper is 64
        x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')

        for i in range(residual_list[0]) :
            x = residual_block(x, channels=ch, is_training=is_training, downsample=True, scope='resblock0_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0')

        for i in range(1, residual_list[1]) :
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0')

        for i in range(1, residual_list[2]) :
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0')

        for i in range(1, residual_list[3]) :
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

        ########################################################################################################


        x = batch_norm(x, is_training, scope='batch_norm')
        x = relu(x)

        x = global_avg_pooling(x)
        x = fully_conneted(x, units=feature_size, scope='logit')

        return 

def resnet18plus(x, is_training=True, reuse=False, res_n = 18, feature_size = 126):
    with tf.variable_scope("network", reuse=reuse):

        if res_n < 50 :
            residual_block = resblock
        else :
            residual_block = bottle_resblock

        residual_list = get_residual_layer(res_n)

        ch = 32 # paper is 64
        x = conv(x, channels=ch, kernel=3, stride=1, scope='conv')  # 384*384

        for i in range(residual_list[0]) :
            x = residual_block(x, channels=ch, is_training=is_training, downsample=False, scope='resblock0_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*2, is_training=is_training, downsample=True, scope='resblock1_0') # 192*192

        for i in range(1, residual_list[1]) :
            x = residual_block(x, channels=ch*2, is_training=is_training, downsample=False, scope='resblock1_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*4, is_training=is_training, downsample=True, scope='resblock2_0') # 96*96

        for i in range(1, residual_list[2]) :
            x = residual_block(x, channels=ch*4, is_training=is_training, downsample=False, scope='resblock2_' + str(i))

        ########################################################################################################

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_3_0') # 48*48

        for i in range(1, residual_list[3]) :
            x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_3_' + str(i))

        ########################################################################################################
        # 48*48*256  
        i = 1
        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_4_0') # 24*24
        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_4_' + str(i))
        # 24*24*256

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_5_0') # 24*24
        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_5_' + str(i))
        # 12*12*256

        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=True, scope='resblock_6_0') # 24*24
        x = residual_block(x, channels=ch*8, is_training=is_training, downsample=False, scope='resblock_6_' + str(i))
        # 6*6*256

        x = batch_norm(x, is_training, scope='batch_norm')
        
        x = tf.reshape(x,[-1, 6*6*256])
        #x = relu(x)

        #x = global_avg_pooling(x)
        #x = fully_conneted(x, units=self.label_dim, scope='logit')

        return x







