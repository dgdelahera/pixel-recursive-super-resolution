import os
import tensorflow as tf
import zipfile

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_path', "data",
                           """Path to the data.""")
tf.app.flags.DEFINE_string('zip_file_path', "data/img_align_celeba.zip",
                           """Path to the zip file.""")
tf.app.flags.DEFINE_string('images_path', "data/images/img_align_celeba",
                           """Path to the photos.""")


def unzip_input(compressed_file, dest_path):
    """ Unzip the zip file with all the images from the dataset.

    Args:
        compressed_file: the file to be unzipped
        dest_path: path where you want to save the pictures
    """
    if os.path.exists(dest_path):
        print('Destination path', dest_path, "already exists.")
        return

    print('Extracting zip content...')
    with zipfile.ZipFile(compressed_file, "r") as zip_ref:
            zip_ref.extractall(dest_path)
    print('All zip content extracted to', dest_path)

def create_datasets(images_path, save_path):
    """ Generates the eval and train datasets from the photos directory. It also creates a file with
        the class names for the labels.

        Args:
            images_path: path to the folder where the input is stored
            save_path: path where the file are being stored
    """
    if os.path.exists(save_path):
        print('Save path', save_path, "already exists.")
        return
    f = open(save_path, 'w')
    for p1 in os.listdir(images_path):
      image = os.path.abspath(images_path + '/' + p1)
      f.write(image + '\n')
    f.close()
    print('Dataset created at', save_path)


def main(none):
    """ Run this function to create the datasets and the numpy array files. """
    unzip_input(FLAGS.zip_file_path, os.path.join(FLAGS.data_path, "images"))
    create_datasets(FLAGS.images_path,  os.path.join(FLAGS.data_path, "train_dataset.txt"))


if __name__ == "__main__":
    tf.app.run()