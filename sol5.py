# ------------- Imports -----------
import scipy.ndimage
import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from keras import layers
from keras import models
from keras import optimizers
import sol5_utils
import matplotlib.pyplot as plt

# ------------- Constants -----------
GRAYSCALE_SHAPE = 2
RGB_SHAPE = 3
GRAY_MAX_VALUE = 255
KERNEL_SIZE = (3, 3)


# ------------- Functions -----------

def read_image(filename, representation):
    """
    3.1 Reading an image into a given representation.
    :param filename: read_image(filename, representation).
    :param representation: representation code, either 1 or 2 defining
    whether the output should be a grayscale image (1) or an RGB image (2).
    If the input image is grayscale, we won’t call it with representation = 2.
    :return: This function returns an image, make sure the output image
    is represented by a matrix of type np.float64 with intensities
    (either grayscale or RGB channel intensities)
    normalized to the range [0, 1].
    """
    im = (imread(filename).astype(np.float64)) / GRAY_MAX_VALUE

    if representation == 1:

        return rgb2gray(im)

    elif representation == 2:

        return (im)


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    Given a set of images, instead of extracting all
    possible image patches and applying a finite set of corruptions,
    we will generate pairs of image patches on the fly, each time picking a
    random image, applying a random corruption 1 , and extracting a random
    patch.
    :param filenames: A list of filenames of clean images.
    :param batch_size: The size of the batch of images for each iteration of
    Stochastic Gradient Descent.
    :param corruption_func: A function receiving a numpy’s array representation
     of an image as a single argument, and returns a randomly corrupted
     version of the input image.
    :param crop_size: ple (height, width) specifying the crop size of the
    patches to extract.
    :return: a Python’s generator object which outputs random tuples of the form.
    """
    images_cache = {}
    source_patches = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
    corrupted_patches = np.zeros((batch_size, crop_size[0], crop_size[1], 1))
    while True:
        # choose randomly a bath of images
        random_indexes = np.random.choice(np.arange(len(filenames)),batch_size)
        # verify which image is already in the dictionary
        for index, randoms in enumerate(random_indexes):
            if filenames[randoms] not in images_cache:
                # read the image with read image func
                im = read_image(filenames[randoms], 1)
                images_cache[filenames[randoms]] = im
            else:
                im = images_cache[filenames[randoms]]

            corrupted_height = crop_size[0] * 3
            corrupted_width = crop_size[1] * 3

            x_patch_start = np.random.randint(0, im.shape[0] - corrupted_height - 1)
            y_patch_start = np.random.randint(0, im.shape[1] - corrupted_width - 1)

            x_patch_end = x_patch_start + corrupted_height
            y_patch_end = y_patch_start + corrupted_width

            patch = im[x_patch_start: x_patch_end,
                    y_patch_start: y_patch_end]

            corrupted_first_patch = corruption_func(patch)

            x_second_patch_start = np.random.randint(0, corrupted_height - crop_size[0] - 1)
            y_second_patch_start = np.random.randint(0, corrupted_width - crop_size[1] - 1)
            x_second_patch_end = x_second_patch_start + crop_size[0]
            y_second_patch_end = y_second_patch_start + crop_size[1]
            patch = patch[x_second_patch_start: x_second_patch_end,
                    y_second_patch_start: y_second_patch_end]

            corrupted_patch = corrupted_first_patch[
                              x_second_patch_start: x_second_patch_end,
                              y_second_patch_start: y_second_patch_end]

            source_patches[index, :, :, 0] = patch - 0.5
            corrupted_patches[index, :, :, 0] = corrupted_patch - 0.5

        yield (corrupted_patches, source_patches)


def resblock(input_tensor, num_channels):
    """
    The above function takes as input a symbolic input tensor and the number of
    channels for each of its convolutional layers, and returns the symbolic
    output tensor of the layer configuration described.
    :param input_tensor: symbolic input tensor.
    :param num_channels: umber of channels for each of its convolutional layers
    :return: the symbolic output tensor of the layer configuration.
    """
    x = layers.Conv2D(num_channels, KERNEL_SIZE, padding="same")(input_tensor)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_channels, KERNEL_SIZE, padding="same")(x)
    x = layers.Add()([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    The above function should return an untrained Keras model, with input
    dimension the shape of (height, width, 1), and all convolutional layers
    (including residual blocks) with number of output channels equal to
    num_channels, except the very last convolutional layer which should have
    a single output channel.
    :param height: The height of the input layer
    :param width: The width of the input layer
    :param num_channels: all convolutional layers with number of output
    channels equal to num_channels
    :param num_res_blocks: The number of residual blocks.
    :return: returns the complete neural network model.
    """
    input_tensor = layers.Input((height, width, 1))
    x = layers.Conv2D(num_channels, KERNEL_SIZE, padding="same")(input_tensor)
    x = layers.Activation('relu')(x)

    for i in range(num_res_blocks):
        x = resblock(x, num_channels)

    x = layers.Conv2D(1, KERNEL_SIZE, padding="same")(x)
    x = layers.Add()([x, input_tensor])
    return models.Model(inputs=input_tensor, outputs=x)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch,
                num_epochs, num_valid_samples):
    """
    This function put everything together and train it.
    :param model: a general neural network model for image restoration.
    :param images: a list of file paths pointing to image files.
    :param corruption_func:
    :param batch_size: the size of the batch of examples for each iteration of SGD.
    :param steps_per_epoch: The number of update steps in each epoch.
    :param num_epochs: The number of epochs for which the optimization will run.
    :param num_valid_samples: The number of samples in the validation set to
    test on after every epoch.
    :return: a trained neural network model on a given training set.
    """
    np.random.shuffle(images)
    dividing_index = np.round(0.8 * len(images)).astype(np.int)
    training_gen = load_dataset(images[:dividing_index], batch_size,
                                corruption_func, model.input_shape[1:3])
    validation_gen = load_dataset(images[dividing_index:], batch_size,
                                  corruption_func, model.input_shape[1:3])
    model.compile(loss='mean_squared_error',
                  optimizer=optimizers.Adam(beta_2=0.9))
    model.fit_generator(generator=training_gen, steps_per_epoch=steps_per_epoch
                        , epochs=num_epochs,
                        validation_steps=num_valid_samples,
                        validation_data=validation_gen)


def restore_image(corrupted_image, base_model):
    """
    The network we learn with train_model can only be used to restore small
    patches. Now we will use this base model to restore full images of any size.
    :param corrupted_image: a grayscale image of shape (height, width)
    and with values in the [0, 1] range of type float64 (as returned by calling
    to read_image from ex1, that is affected by a corruption generated from the
    same corruption function encountered during training (the image is not
    necessarily from the training set though). You can assume the size of the
    image is at least as large as the size of the image patches during training.
    :param base_model: a neural network trained to restore small patches
    (the model described in section 4, after being trained in section 5).
    The input and output of the network are images with values in the
    [−0.5, 0.5] range (remember we subtracted 0.5 in the dataset generation).
    You should take this into account when preprocessing the images.
    :return:
    """
    # You will need to create a new model that fits the size of the input
    # image and has the same weights as the given base model.
    a = layers.Input((corrupted_image.shape[0], corrupted_image.shape[1], 1))
    b = base_model(a)
    new_model = models.Model(inputs=a, outputs=b)
    # adding dimension to the corrupted image
    corrupted_image = corrupted_image.reshape((1, corrupted_image.shape[0], corrupted_image.shape[1], 1))
    corrupted_image -= 0.5
    res = new_model.predict(corrupted_image)
    res += 0.5
    res = np.clip(res, a_min=0, a_max=1)
    return res[0, :, :, 0].astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Random noise function for training and then testing my model.
    :param image: a grayscale image with values in the [0, 1] range of type
    float64.
    :param min_sigma: a non-negative scalar value representing the minimal
     variance of the gaussian distribution.
    :param max_sigma: a non-negative scalar value larger than or equal to
     min_sigma, representing the maximal variance of the gaussian distribution.
    :return: noised image
    """
    # Which should randomly sample a value of sigma, uniformly distributed
    # between min_sigma and max_sigma
    sigma = min_sigma + (np.random.random_sample() * (max_sigma - min_sigma))
    # adding to every pixel of the input image a zero-mean gaussian random
    # variable with standard deviation equal to sigma.
    im = image + np.random.normal(size=image.shape, scale=sigma)
    # Before returning the results, the values should be rounded
    # to the nearest fraction i / 255 and clipped to [0,1].
    im *= 255
    im /= 255
    return im.clip(min=0, max=1)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    This function return a trained denoising model given the function -
    add_gaussian_noise.
    :param num_res_blocks: The number of residual blocks.
    :param quick_mode: quick mode for the above function for testing purposes.
    :return: A trained model
    """
    images_for_denoising = sol5_utils.images_for_denoising()
    corruption_func = lambda x: add_gaussian_noise(x, 0, 0.2)
    model = build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30
    else:
        batch_size = 100
        steps_per_epoch = 100
        num_epochs = 5
        num_valid_samples = 1000

    train_model(model, images_for_denoising, corruption_func,
                batch_size,
                steps_per_epoch, num_epochs, num_valid_samples)

    return model


def add_motion_blur(image, kernel_size, angle):
    """
    Motion blur can be simulated by convolving the image with a kernel made of
    a single line crossing its center, where the direction of the motion blur
    is given by the angle of the line.
    :param image: a grayscale image with values in the [0, 1] range of type
    float64.
    :param kernel_size: an odd integer specifying the size of the
    kernel (even integers are ill-defined).
    :param angle: an angle in radians in the range [0, π).
    :return: a blured image.
    """
    blur_kernel = sol5_utils.motion_blur_kernel(kernel_size, angle)
    im = scipy.ndimage.filters.convolve(image, blur_kernel)
    im *= 255
    im /= 255
    return im.clip(min=0, max=1)


def random_motion_blur(image, list_of_kernel_sizes):
    """
    This function samples an angle at uniform from the range [0, π),
    and choses a kernel size at uniform from the list list_of_kernel_sizes,
    followed by applying the previous function with the given image and the
    randomly sampled parameters.
    :param image: a grayscale image with values in the [0, 1] range of type
    float64.
    :param list_of_kernel_sizes: a list of odd integers.
    :return:
    """
    angle = np.random.random_sample() * (np.pi)
    kernel_size = np.random.choice(list_of_kernel_sizes)
    return add_motion_blur(image, kernel_size, angle)

def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    function which will return a trained deblurring model.
    :param num_res_blocks:
    :param quick_mode:
    :return: a trained deblurring model.
    """
    images_for_denoising = sol5_utils.images_for_deblurring()
    corruption_func = lambda x: random_motion_blur(x, [7])
    model = build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        batch_size = 10
        steps_per_epoch = 3
        num_epochs = 2
        num_valid_samples = 30
    else:
        batch_size = 100
        steps_per_epoch = 100
        num_epochs = 10
        num_valid_samples = 1000

    train_model(model, images_for_denoising, corruption_func,
                batch_size,
                steps_per_epoch, num_epochs, num_valid_samples)
    return model
