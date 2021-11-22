# digital networks for End-to-end optimization
# Author: Yicheng Wu @ Rice University
# 03/29/2019


import tensorflow as tf


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv2dPad(x, W):
    N = int((W.shape[0].value - 1) / 2)
    x_pad = tf.pad(x, [[0, 0], [N, N], [N, N], [0, 0]], "SYMMETRIC")
    return tf.nn.conv2d(x_pad, W, strides=[1, 1, 1, 1], padding='VALID')


def BN(x, phase_BN, scope):
    return tf.layers.batch_normalization(x, momentum=0.9, training=phase_BN)


def cnnLayerPad(scope_name, inputs, outChannels, kernel_size, is_training, relu=True, maxpool=True):
    with tf.variable_scope(scope_name) as scope:
        inChannels = inputs.shape[-1].value
        W_conv = tf.get_variable('W_conv', [kernel_size, kernel_size, inChannels, outChannels])
        b_conv = tf.get_variable('b_conv', [outChannels])
        x_conv = conv2dPad(inputs, W_conv) + b_conv
        out = BN(x_conv, is_training, scope)
        if relu:
            out = tf.nn.relu(out)
        if maxpool:
            out = max_pool_2x2(out)
        return out


def cnn3x3Pad(scope_name, inputs, outChannels, is_training, relu=True):
    with tf.variable_scope(scope_name) as scope:
        inChannels = inputs.shape[-1].value
        W_conv = tf.get_variable('W_conv', [3, 3, inChannels, outChannels])
        b_conv = tf.get_variable('b_conv', [outChannels])
        x_conv = conv2dPad(inputs, W_conv) + b_conv
        out = BN(x_conv, is_training, scope)
        if relu:
            out = tf.nn.relu(out)

        return out


def deconv(scope_name, inputs, outChannels):
    with tf.variable_scope(scope_name) as scope:
        inChannels = inputs.shape[-1].value
        Nx = inputs.shape[1].value
        Ny = inputs.shape[2].value
        W = tf.get_variable('resize_conv', [3, 3, inChannels, outChannels])
        resize = tf.image.resize_nearest_neighbor(inputs, [2 * Nx, 2 * Ny])
        output = conv2dPad(resize, W)
        return output


def UNet(inputs, phase_BN):
    # in the decoder, instead of using transpose conv, do resize by 2 + conv2
    # for each conv, do pad+valid instead of same
    down1_1 = cnn3x3Pad('down1_1', inputs, 32, phase_BN)
    down1_2 = cnn3x3Pad('down1_2', down1_1, 32, phase_BN)

    down2_0 = max_pool_2x2(down1_2)
    down2_1 = cnn3x3Pad('down2_1', down2_0, 64, phase_BN)
    down2_2 = cnn3x3Pad('down2_2', down2_1, 64, phase_BN)

    down3_0 = max_pool_2x2(down2_2)
    down3_1 = cnn3x3Pad('down3_1', down3_0, 128, phase_BN)
    down3_2 = cnn3x3Pad('down3_2', down3_1, 128, phase_BN)

    down4_0 = max_pool_2x2(down3_2)
    down4_1 = cnn3x3Pad('down4_1', down4_0, 256, phase_BN)
    down4_2 = cnn3x3Pad('down4_2', down4_1, 256, phase_BN)

    down5_0 = max_pool_2x2(down4_2)
    down5_1 = cnn3x3Pad('down5_1', down5_0, 512, phase_BN)
    down5_2 = cnn3x3Pad('down5_2', down5_1, 512, phase_BN)

    up4_0 = tf.concat([deconv('up4_0', down5_2, 256), down4_2], axis=-1)
    up4_1 = cnn3x3Pad('up4_1', up4_0, 256, phase_BN)
    up4_2 = cnn3x3Pad('up4_2', up4_1, 256, phase_BN)

    up3_0 = tf.concat([deconv('up3_0', up4_2, 128), down3_2], axis=-1)
    up3_1 = cnn3x3Pad('up3_1', up3_0, 128, phase_BN)
    up3_2 = cnn3x3Pad('up3_2', up3_1, 128, phase_BN)

    up2_0 = tf.concat([deconv('up2_0', up3_2, 64), down2_2], axis=-1)
    up2_1 = cnn3x3Pad('up2_1', up2_0, 64, phase_BN)
    up2_2 = cnn3x3Pad('up2_2', up2_1, 64, phase_BN)

    up1_0 = tf.concat([deconv('up1_0', up2_2, 32), down1_2], axis=-1)
    up1_1 = cnn3x3Pad('up1_1', up1_0, 32, phase_BN)
    up1_2 = cnn3x3Pad('up1_2', up1_1, 32, phase_BN)

    up1_3 = cnnLayerPad('up1_3', up1_2, 1, 1, phase_BN, relu=False, maxpool=False)
    out = tf.tanh(up1_3) + inputs

    # set the range to be in (0,1)
    out = tf.minimum(out, 1.0)
    out = tf.maximum(out, 0.0)

    return out

