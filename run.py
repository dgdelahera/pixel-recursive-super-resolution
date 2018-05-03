
import numpy as np
import tensorflow as tf

from skimage.io import imsave
from skimage.transform import resize


from utils import logits_2_pixel_value
from net import Net

from style_transfer.conditional_instance_norm.cin.utils.flags import Flags
from style_transfer.conditional_instance_norm.cin.utils.image_utils import im_read

args = Flags()


class SuperResModel(object):

    def __init__(self, net, low_res_image, high_res_image):
        self.low_res_image = low_res_image
        self.high_res_image = high_res_image
        self.net = net


def add_arguments():
    args.add_argument('--input_image', type=str, default=None,
                      help='Path of low resolution image.')

    args.add_argument('--out_image_path', type=str, default=None,
                      help='Path to the high resolution output.')

    args.add_argument('--ckpt', type=str, default=None,
                      help='Path to model checkpoint.')


def load_image(image_path):

    image = im_read(image_path)
    image = np.ndarray.astype(image, dtype=np.float32)

    return image


def save_image(image, image_path):

    image = np.uint8(image)
    imsave(image_path, image)


def create_network(low_res_shape, high_res_shape):

    low_res_image = tf.placeholder(dtype=tf.float32, shape= low_res_shape, name='low_res_image')
    high_res_image = tf.placeholder(dtype=tf.float32, shape=high_res_shape, name='high_res_image')
    net = Net(hr_images=high_res_image, lr_images=low_res_image, scope='prsr')

    return SuperResModel(net=net, low_res_image=low_res_image, high_res_image=high_res_image)


def enhance(model, low_res_input, high_res_shape, mu=1.1):

    high_res_out_image = np.zeros(shape=high_res_shape, dtype=np.float32)

    c_logits = model.net.conditioning_logits
    p_logits = model.net.prior_logits

    with tf.Session() as session:

        # Restore checkpoint.
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(session, args.ckpt)

        np_c_logits = session.run(
            c_logits,
            feed_dict={model.low_res_image: low_res_input, model.net.train: False}
        )

        batch_size, height, width, channels = high_res_shape

        for i in range(height):
            for j in range(width):
                for c in range(channels):
                    np_p_logits = session.run(
                        p_logits,
                        feed_dict={model.high_res_image: high_res_out_image}
                    )

                    sum_logits = np_c_logits[:, i, j, c * 256:(c + 1) * 256] + \
                                 np_p_logits[:, i, j, c * 256:(c + 1) * 256]

                    new_pixel = logits_2_pixel_value(sum_logits, mu=mu)
                    high_res_out_image[:, i, j, c] = new_pixel

        return high_res_out_image


def main():

    low_res_input = load_image(args.input_image)

    low_res_input = np.expand_dims(low_res_input, axis=0)

    batch_size, height, width, channels = low_res_input.shape
    out_shape = [batch_size, height * 4, width * 4, channels]

    model = create_network(list(low_res_input.shape), out_shape)

    out_high_res = enhance(
        model,
        low_res_input=low_res_input,
        high_res_shape=out_shape,
    )

    save_image(out_high_res[0], args.out_image_path)


if __name__ == '__main__':
    add_arguments()
    main()