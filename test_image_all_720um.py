import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import scipy.misc
import Network
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # only uses GPU 1


results_dir = './results/'
results_best_dir = './results/best_model/'


save_file_UNET = './results/npo_cubic_e2e_MSE.csv'
save_file_UNET_SSIM = './results/npo_cubic_e2e_SSIM.csv'


# read test image
image_read_dir = './image/'
GT_path = str(Path(image_read_dir + 'test_img.png'))
gt_img = cv2.imread(GT_path, 0)/255
GT = np.tile(gt_img, [21, 1, 1])
GT = np.expand_dims(GT, -1)
GT = tf.convert_to_tensor(GT, dtype=tf.float32)

#save image
image_save_dir = './image/results/'

####### read from the TFRECORD format #################
## for faster reading from Hard disk
def read_tfrecord(TFRECORD_PATH):
    # from tfrecord file to data
    N_w = 326  # size of the images
    N_h = 326
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
    N_raw = 326  # for boundary effect, 256+70, will need cropping after convolution

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
    #RGB_batch = tf.train.shuffle_batch([RGB], batch_size=batchsize, capacity=200, num_threads=5)
    RGB = tf.expand_dims(RGB, axis=0)
    RGB_batch = tf.tile(RGB, [21,1,1,1])
    RGB_batch_float = tf.image.convert_image_dtype(RGB_batch, tf.float32)

    # padd the target for convolution
    RGB_batch_float = tf.image.resize_image_with_crop_or_pad(RGB_batch_float, 298, 298)

    return RGB_batch_float[:, :, :, 0:1]



def add_gaussian_noise(images, std):
    noise = tf.random_normal(shape=tf.shape(images), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.nn.relu(images + noise)


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
def system(PSFs, RGB_batch_float, phase_BN=False):
    with tf.variable_scope("system", reuse=tf.AUTO_REUSE):
        blur = blurImage_diffPatch_diffDepth(RGB_batch_float, PSFs)  # size [batch_size * N_Phi, Nx, Ny, 3]

        # noise
        sigma = 0.01
        blur_noisy = add_gaussian_noise(blur, sigma)

        RGB_hat = Network.UNet(blur_noisy, phase_BN)

        return blur_noisy, RGB_hat


######################  RMS cost #############################
## @param{GT}: ground truth
## @param{hat}: reconstruction
##############################################################
def cost_rms(GT, hat):
    cost = tf.sqrt(tf.reduce_mean(tf.reduce_mean((tf.square(GT - hat)),1),1))
    return cost

######################  SSIM cost #############################
## @param{GT}: ground truth
## @param{hat}: reconstruction
##############################################################
def cost_ssim(GT, hat):
    cost = tf.image.ssim(GT, hat, 1.0) # assume img intensity ranges from 0 to 1
    cost = tf.expand_dims(cost, axis = 1)
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
    blur_GT = blur[:, int((N_B - 1) / 2):-int((N_B - 1) / 2), int((N_B - 1) / 2):-int((N_B - 1) / 2),
              :]  # crop the patch to 256x256

    cost = tf.sqrt(tf.reduce_mean(tf.square(blur_GT - reblur)))

    return reblur, cost


########################################################  PARAMETER ####################################################
N_B = 71

N_Phi = 21
batch_size = N_Phi
Phi_list = np.linspace(-10, 10, N_Phi, np.float32)  # defocus

PSFs = np.load(results_dir + 'PSFs.npy')
PSFs = tf.convert_to_tensor(PSFs, dtype=tf.float32)


####################################################### architecture ###################################################
RGB_batch_float_test = GT

[blur_test, RGB_hat_test] = system(PSFs, RGB_batch_float_test)

# cost function
with tf.name_scope("cost"):
    RGB_GT_test = RGB_batch_float_test[:, int((N_B - 1) / 2):-int((N_B - 1) / 2),
                  int((N_B - 1) / 2):-int((N_B - 1) / 2), :]  # crop the all-in-focus to be


    cost_rms_test = cost_rms(RGB_GT_test, RGB_hat_test)
    cost_ssim_test = cost_ssim(RGB_GT_test, RGB_hat_test)



###################################################  reload model ##################################################
saver_best = tf.train.Saver()

with tf.Session() as sess:

    # threading for parallel
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    model_path = tf.train.latest_checkpoint(results_dir)
    saver_best.restore(sess, model_path)

    for i in range(1):
        [loss_blur_test, loss_estimate_test, loss_rms_test, loss_ssim_test, GT_test] = sess.run(
        	[blur_test, RGB_hat_test, cost_rms_test, cost_ssim_test, RGB_GT_test])

        sharp_crop = GT_test[0, :, :, 0]
        gt_min = np.amin(np.ndarray.flatten(sharp_crop))
        gt_max = np.amax(np.ndarray.flatten(sharp_crop))
        scipy.misc.toimage(sharp_crop, cmin=gt_min, cmax=gt_max).save(image_save_dir + 'sharp_crop.png')

        np.savetxt(save_file_UNET, loss_rms_test, delimiter=',', newline='\n')
        np.savetxt(save_file_UNET_SSIM, loss_ssim_test, delimiter=',', newline='\n')

    np.save('blur.npy', loss_blur_test)
    np.save('estimate.npy', loss_estimate_test)


    coord.request_stop()
    coord.join(threads)

print('Now saving the images')

mask_blur = np.load('blur.npy')
estimate = np.load('estimate.npy')


def npy_to_images(npy_stack, save_name):
    for i in range(21):
        img_cur = npy_stack[i, :, :, 0]
        img_cur = 1 - img_cur
        img_min = np.amin(np.ndarray.flatten(img_cur))
        img_max = np.amax(np.ndarray.flatten(img_cur))
        save_name_cur = '00_'+ str(i) + '_' + save_name
        scipy.misc.toimage(img_cur, cmin=img_min, cmax=img_max).save(image_save_dir + save_name_cur)


npy_to_images(mask_blur, 'deepDOF_blur.png')
npy_to_images(estimate, 'deepDOF_hat.png')
