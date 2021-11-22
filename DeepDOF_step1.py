# End-to-end optimization for EDOF
# Author: Yicheng Wu @ Rice University
# 03/29/2019
# 04/12/2019 parameter update
# 11/7/2019 update best model with valid_loss
# 12/3/2019 update best model with valid_rms instead of valid_loss
# 12/3/2019 update reblur cost = rms(blur, reblur)

import tensorflow as tf
import scipy.io as sio
import numpy as np
import os
import Network

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"            # only uses GPU 1

results_dir = "./results/"
DATA_PATH = "./DATA/"
TFRECORD_TRAIN_PATH = [DATA_PATH + 'npo_720um_train.tfrecords']  # for testing purpose both are validation sets
TFRECORD_VALID_PATH = [DATA_PATH + 'npo_720um_train.tfrecords']

## optimizer learning rates
# use 0 in step 1:
lr_optical = 0
# use 1e-9 in step 2:
# lr_optical = 1e-9
lr_digital = 1e-4
print('lr_optical:' + str(lr_optical))
print('lr_digital:' + str(lr_digital))


##########################################   Functions  #############################################

# Peak SNR, could be used as cost function
def tf_PSNR(a, b, max_val, name=None):
    with tf.name_scope(name, 'PSNR', [a, b]):
        # Need to convert the images to float32.  Scale max_val accordingly so that
        # PSNR is computed correctly.
        max_val = tf.cast(max_val, tf.float32)
        a = tf.cast(a, tf.float32)
        b = tf.cast(b, tf.float32)
        mse = tf.reduce_mean(tf.squared_difference(a, b), [-3, -2, -1])
        psnr_val = tf.subtract(
            20 * tf.log(max_val) / tf.log(10.0),
            np.float32(10 / np.log(10)) * tf.log(mse),
            name='psnr')

        return psnr_val


####### read from the TFRECORD format #################
## for faster reading from Hard disk
def read_tfrecord(TFRECORD_PATH):
    # from tfrecord file to data
    N_w = 1000 # size of the images
    N_h = 1000
    queue = tf.train.string_input_producer(TFRECORD_PATH, shuffle=True)
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(queue)  

    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'sharp': tf.FixedLenFeature([], tf.string),
                                       })

    RGB_flat = tf.decode_raw(features['sharp'], tf.uint8)
    RGB = tf.reshape(RGB_flat, [N_h, N_w, 1]) 

    return RGB



########## Preprocess the images #############
##  crop to patches
##  random flip
##  Add uniform noise
############################################  
def data_augment(RGB_batch_float):
    # crop to N_raw x N_raw
    N_raw = 326 # for boundary effect, 256+70, will need cropping after convolution
    data1 = tf.map_fn(lambda img: tf.random_crop(img, [N_raw, N_raw, 1]), RGB_batch_float)

    # flip both images and labels
    data2 = tf.map_fn(lambda img: tf.image.random_flip_up_down(tf.image.random_flip_left_right(img)), data1)

    # only adjust the RGB value of the image
    r1 = tf.random_uniform([]) * 0.3 + 0.8
    RGB_out = data2 * r1

    return RGB_out



############ Put data in batches #############
##  put in batch and shuffle
##  cast to float32
##  call data_augment for image preprocess
## @param{TFRECORD_PATH}: path to the data
## @param{batchsize}: currently 21 for the 21 PSFs
##############################################
def read2batch(TFRECORD_PATH, batchsize):
    # load tfrecord and make them to be usable data
    RGB = read_tfrecord(TFRECORD_PATH)
    RGB_batch = tf.train.shuffle_batch([RGB], batch_size=batchsize, capacity=200,
                                       min_after_dequeue=50, num_threads=5)
    RGB_batch_float = tf.image.convert_image_dtype(RGB_batch, tf.float32)

    RGB_batch_float = data_augment(RGB_batch_float)

    return RGB_batch_float[:,:,:,0:1]


def add_gaussian_noise(images, std):
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.nn.relu(images + noise)




########### fftshift2D ###################
## the same as fftshift in MATLAB
## works for complex number
def fft2dshift(input):
    dim = int(input.shape[1].value)  # dimension of the data
    channel1 = int(input.shape[0].value)  # channels for the first dimension
    if dim % 2 == 0:
        # even version
        # shift up and down
        u = tf.slice(input, [0, 0, 0], [channel1, int((dim) / 2), dim])
        d = tf.slice(input, [0, int((dim) / 2), 0], [channel1, int((dim) / 2), dim])
        du = tf.concat([d, u], axis=1)
        # shift left and right
        l = tf.slice(du, [0, 0, 0], [channel1, dim, int((dim) / 2)])
        r = tf.slice(du, [0, 0, int((dim) / 2)], [channel1, dim, int((dim) / 2)])
        output = tf.concat([r, l], axis=2)
    else:
        # odd version
        # shift up and down
        u = tf.slice(input, [0, 0, 0], [channel1, int((dim + 1) / 2), dim])
        d = tf.slice(input, [0, int((dim + 1) / 2), 0], [channel1, int((dim - 1) / 2), dim])
        du = tf.concat([d, u], axis=1)
        # shift left and right
        l = tf.slice(du, [0, 0, 0], [channel1, dim, int((dim + 1) / 2)])
        r = tf.slice(du, [0, 0, int((dim + 1) / 2)], [channel1, dim, int((dim - 1) / 2)])
        output = tf.concat([r, l], axis=2)
    return output



#########  generate out-of-focus phase  ###############
## @param{Phi_list}: a list of Phi values
## @param{N_B}: size of the blur kernel
## @return{OOFphase} 
def gen_OOFphase(Phi_list, N_B):
    # return (Phi_list,pixel,pixel,color)
    N = N_B
    x0 = np.linspace(-2.84, 2.84, N) # 71/25 =2.84
    xx, yy = np.meshgrid(x0, x0)
    OOFphase = np.empty([len(Phi_list), N, N, 1], dtype=np.float32)
    for j in range(len(Phi_list)):
        Phi = Phi_list[j]
        OOFphase[j, :, :, 0] = Phi * (xx ** 2 + yy ** 2)
    return OOFphase



##################  Generates the PSFs  ########################
## @param{h}: height map of the mask
## @param{OOFphase}: out-of-focus phase
## @param{wvls}: wavelength \lambda
## @param{idx}: index of the PSF
## @param{N_B}: size of the blur kernel
#################################################################
def gen_PSFs(h, OOFphase, wvls, idx, N_B):
    n = 1.5  # diffractive index

    with tf.variable_scope("PSFs"):
        OOFphase_B = OOFphase[:, :, :, 0]
        phase_B = tf.add(2 * np.pi / wvls[0] * (n - 1) * h, OOFphase_B) # phase modulation of mask (phi_M)
        Pupil_B = tf.multiply(tf.complex(idx, 0.0), tf.exp(tf.complex(0.0, phase_B)), name='Pupil_B') # pupil P
        Norm_B = tf.cast(N_B * N_B * np.sum(idx ** 2), tf.float32)  # what's this?
        PSF_B = tf.divide(tf.square(tf.abs(fft2dshift(tf.fft2d(Pupil_B)))), Norm_B, name='PSF_B')

    return tf.expand_dims(PSF_B, -1)



################  blur the images using PSFs  ##################
## same patch different depths put in a stack
################################################################
def one_wvl_blur(im, PSFs0):
    N_B = PSFs0.shape[1].value
    N_Phi = PSFs0.shape[0].value
    N_im = im.shape[1].value
    N_im_out = N_im - N_B + 1  # the final image size after blurring

    sharp = tf.transpose(tf.reshape(im, [-1, N_Phi, N_im, N_im]),
                         [0, 2, 3, 1])  # reshape to make N_Phi in the last channel
    PSFs = tf.expand_dims(tf.transpose(PSFs0, perm=[1, 2, 0]), -1)
    blurAll = tf.nn.depthwise_conv2d(sharp, PSFs, strides=[1, 1, 1, 1], padding='VALID')
    blurStack = tf.transpose(
        tf.reshape(tf.transpose(blurAll, perm=[0, 3, 1, 2]), [-1, 1, N_im_out, N_im_out]),
        perm=[0, 2, 3, 1])  # stack all N_Phi images to the first dimension

    return blurStack


def blurImage_diffPatch_diffDepth(RGB, PSFs):
    blur = one_wvl_blur(RGB[:, :, :, 0], PSFs[:, :, :, 0])

    return blur


####################### system ##########################
## @param{PSFs}: the PSFs
## @param{RGB_batch_float}: patches
## @param{phase_BN}: batch normalization, True only during training
########################################################
def system(PSFs, RGB_batch_float, phase_BN=True): 
    with tf.variable_scope("system", reuse=tf.AUTO_REUSE):
        blur = blurImage_diffPatch_diffDepth(RGB_batch_float, PSFs)  # size [batch_size * N_Phi, Nx, Ny, 3]

        # noise
        sigma = 0.01
        blur_noisy = add_gaussian_noise(blur, sigma)

        RGB_hat = Network.UNet(blur_noisy, phase_BN)

        return blur, RGB_hat


######################  RMS cost #############################
## @param{GT}: ground truth
## @param{hat}: reconstruction
##############################################################
def cost_rms(GT, hat):
    cost = tf.sqrt(tf.reduce_mean(tf.square(GT - hat)))
    return cost


##########  compare the reconstruction reblured with U-net input?  ############
## important for EDOF to utilize the PSF information
## @param{RGB_hat}: Unet reconstructed image
## @param{PSFs}: PSF used
## @param{blur}: all-in-focus image conv PSF
## @param{N_B}: size of blur kernel
## @return{reblur}: reconstruction blurred
## @return{cost}: l2 norm between blur_GT and reblur
##############################################################################
def cost_reblur(RGB_hat, PSFs, blur, N_B):
    reblur = blurImage_diffPatch_diffDepth(RGB_hat, PSFs)
    blur_GT = blur[:, int((N_B - 1) / 2):-int((N_B - 1) / 2), int((N_B - 1) / 2):-int((N_B - 1) / 2), :] #crop the patch to 256x256

    cost = tf.sqrt(tf.reduce_mean(tf.square(blur_GT - reblur)))

    return reblur, cost


######################################### Set parameters   ###############################################

# def main():

zernike = sio.loadmat('zernike_basis_150mm.mat')
u2 = zernike['u2']  # basis of zernike poly
idx = zernike['idx']
idx = idx.astype(np.float32)

a_zernike_mat = sio.loadmat('a_zernike_cubic_150mm.mat')
a_zernike_fix = a_zernike_mat['a']
a_zernike_fix = a_zernike_fix * 4
a_zernike_fix = tf.convert_to_tensor(a_zernike_fix)

N_B = 71  # size of the blur kernel
wvls = np.array([550]) * 1e-9 # wavelength 550 nm
N_color = len(wvls)

N_modes = u2.shape[1]  # load zernike modes

# generate the defocus phase
N_Phi = 21
Phi_list = np.linspace(-10, 10, N_Phi, np.float32) # defocus
OOFphase = gen_OOFphase(Phi_list, N_B)  # return (N_Phi,N_B,N_B,N_color)

# baseline offset for the heightmap
c = 0

####################################   Build the architecture  #####################################################


with tf.variable_scope("PSFs"):
    a_zernike_learn = tf.get_variable("a_zernike_learn", [N_modes, 1], initializer=tf.zeros_initializer(),
                                constraint=lambda x: tf.clip_by_value(x, -wvls[0] / 2, wvls[0] / 2))
    a_zernike = a_zernike_learn + a_zernike_fix # fixed cubic and learning part
    g = tf.matmul(u2, a_zernike)
    h = tf.nn.relu(tf.reshape(g, [N_B, N_B])+c, # c: baseline
                   name='heightMap')  # height map of the phase mask, should be all positive
    PSFs = gen_PSFs(h, OOFphase, wvls, idx, N_B)  # return (N_Phi, N_B, N_B, N_color)


batch_size = N_Phi  # it means that each patch is blurred at different depth. Will be an error if this is not N_Phi


RGB_batch_float = read2batch(TFRECORD_TRAIN_PATH, batch_size)
RGB_batch_float_valid = read2batch(TFRECORD_VALID_PATH, batch_size)

[blur_train, RGB_hat_train] = system(PSFs, RGB_batch_float)
[blur_valid, RGB_hat_valid] = system(PSFs, RGB_batch_float_valid, phase_BN=False)

# cost function
with tf.name_scope("cost"):
    RGB_GT_train = RGB_batch_float[:, int((N_B - 1) / 2):-int((N_B - 1) / 2),
                   int((N_B - 1) / 2):-int((N_B - 1) / 2), :]                    # crop the all-in-focus to be 
    RGB_GT_valid = RGB_batch_float_valid[:, int((N_B - 1) / 2):-int((N_B - 1) / 2),
                   int((N_B - 1) / 2):-int((N_B - 1) / 2), :]

    cost_rms_train = cost_rms(RGB_GT_train, RGB_hat_train)
    cost_rms_valid = cost_rms(RGB_GT_valid, RGB_hat_valid)

    cost_train = cost_rms_train
    cost_valid = cost_rms_valid

# train ditial and optical part saparetely
vars_optical = tf.trainable_variables("PSFs")
vars_digital = tf.trainable_variables("system")

opt_optical = tf.train.AdamOptimizer(lr_optical)
opt_digital = tf.train.AdamOptimizer(lr_digital)

global_step = tf.Variable(0, name='global_step', trainable=False)  # initialize the stepsize

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # update the variables with gradient descent
with tf.control_dependencies(update_ops):
    grads = tf.gradients(cost_train, vars_optical + vars_digital)
    grads_optical = grads[:len(vars_optical)]
    grads_digital = grads[len(vars_optical):]
    train_op_optical = opt_optical.apply_gradients(zip(grads_optical, vars_optical))
    train_op_digital = opt_digital.apply_gradients(zip(grads_digital, vars_digital))
    train_op = tf.group(train_op_optical, train_op_digital)

# tensorboard
tf.summary.scalar('cost_train', cost_train)
tf.summary.scalar('cost_valid', cost_valid)
tf.summary.scalar('cost_rms_train', cost_rms_train)
tf.summary.scalar('cost_rms_valid', cost_rms_valid)

tf.summary.histogram('a_zernike', a_zernike)
tf.summary.histogram('a_zernike_learn', a_zernike_learn)
tf.summary.histogram('a_zernike_fix', a_zernike_fix)
tf.summary.image('Height', tf.expand_dims(tf.expand_dims(h, 0), -1))
tf.summary.image('sharp_valid', tf.image.convert_image_dtype(RGB_GT_valid[0:1, :, :, :], dtype = tf.uint8))
tf.summary.image('blur_valid', tf.image.convert_image_dtype(blur_valid[0:1, :, :, :], dtype = tf.uint8))
tf.summary.image('RGB_hat_valid', tf.image.convert_image_dtype(RGB_hat_valid[0:1, :, :, :], dtype = tf.uint8))
tf.summary.image('PSF_n100', PSFs[0:1,:,:])
tf.summary.image('PSF_p90', PSFs[19:20,:,:])
tf.summary.image('PSF_p100', PSFs[20:21,:,:])
merged = tf.summary.merge_all()

##########################################   Train  #############################################

# variables_to_restore = [v for v in tf.global_variables() if v.name.startswith('system')]
# saver = tf.train.Saver(variables_to_restore)
saver_all = tf.train.Saver(max_to_keep=1)
saver_best = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    best_dir = 'best_model/'
    if not os.path.exists(results_dir + best_dir):
        os.makedirs(results_dir + best_dir)
        best_valid_loss = 100
    else:
        best_valid_loss = np.loadtxt(results_dir + 'best_valid_loss.txt')
        print('Current best valid loss = ' + str(best_valid_loss))

    if not tf.train.checkpoint_exists(results_dir + 'checkpoint'):
        # option1: run a new one
        out_all = np.empty((0, 2))  # for out_all 4D: [train_loss,valid_loss,train_acc,valid_acc]
        print('Start to save at: ', results_dir)
    else:
        print(results_dir)
        model_path = tf.train.latest_checkpoint(results_dir)
        load_path = saver_all.restore(sess, model_path)
        out_all = np.load(results_dir + 'out_all.npy')
        print('Continue to save at: ', results_dir)

    train_writer = tf.summary.FileWriter(results_dir + '/summary/', sess.graph)

    # threading for parallel 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(100000):
        ## load the batch
        train_op.run()  # only train digital 

        if i != 0 and i % 10000 == 0:
            lr_digital = lr_digital / 5  # reduce the learning rate every 10k

        if i % 10 == 0:
            [train_summary, loss_train, loss_valid, loss_rms_valid] = sess.run(
                [merged, cost_train, cost_valid, cost_rms_valid])
            train_writer.add_summary(train_summary, i)

            print("Iter " + str(i) + ", Train Loss = " + \
                  "{:.6f}".format(loss_train) + ", Valid Loss = " + \
                  "{:.6f}".format(loss_valid))

            # save them
            out = np.array([[loss_train, loss_valid]])
            out_all = np.vstack((out_all, out))
            np.save(results_dir + 'out_all.npy', out_all)

            saver_all.save(sess, results_dir + "model.ckpt", global_step=i)

            [ht, at, PSFst] = sess.run([h, a_zernike, PSFs])
            np.savetxt(results_dir + 'HeightMap.txt', ht)
            np.savetxt(results_dir + 'a_zernike.txt', at)
            np.save(results_dir + 'PSFs.npy', PSFst)

            if (loss_rms_valid < best_valid_loss) and (i > 1):
                best_valid_loss = loss_rms_valid
                np.savetxt(results_dir + 'best_valid_loss.txt', [best_valid_loss])
                saver_best.save(sess, results_dir + best_dir + "model.ckpt")
                np.save(results_dir + best_dir + 'out_all.npy', out_all)
                np.savetxt(results_dir + best_dir + 'HeightMap.txt', ht)
                print('best at iter ' + str(i) + ' with loss = ' + str(best_valid_loss))

    train_writer.close()
    coord.request_stop()
    coord.join(threads)
