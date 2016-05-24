from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True
import tensorflow as tf
import numpy as np
import os.path
from caffe_net import CaffeNet
from scipy import misc

'''load the two caffenets from npy files'''
def load_caffenets(sess, cpu=True):
    '''set device'''
    if cpu:
        with tf.device('/cpu'):
            original='caffe_net_original.npy'
            opticalflow='caffe_net_opticalflow.npy'
            original_input = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
            opticalflow_input = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
            net_original = CaffeNet({'data': original_input}, prefix='original_')
            net_opticalflow = CaffeNet({'data': opticalflow_input}, prefix='opticalflow_')
            original_output = net_original.layers['prob'] #softmax output
            flow_output = net_opticalflow.layers['prob'] #softmax output
            combined_output = tf.mul(original_output, flow_output) #per-element product

            print('Loading the model')
            net_original.load(original, sess, True)
            net_opticalflow.load(opticalflow, sess, True)
            return original_output, flow_output, combined_output, original_input, opticalflow_input
    else:
        original='caffe_net_original.npy'
        opticalflow='caffe_net_opticalflow.npy'
        original_input = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
        opticalflow_input = tf.placeholder(tf.float32, shape=(None, 227, 227, 3))
        net_original = CaffeNet({'data': original_input}, prefix='original_')
        net_opticalflow = CaffeNet({'data': opticalflow_input}, prefix='opticalflow_')
        original_output = net_original.layers['prob'] #softmax output
        flow_output = net_opticalflow.layers['prob'] #softmax output
        combined_output = tf.mul(original_output, flow_output) #per-element product

        print('Loading the model')
        net_original.load(original, sess, True)
        net_opticalflow.load(opticalflow, sess, True)
        return original_output, flow_output, combined_output, original_input, opticalflow_input

'''
get file names lists
return original list, flow list, and label list
'''
def get_filenames():
    original_data_dir = '/shared/original/thumbs'
    opticalflow_data_dir = '/shared/flow/thumbs'
    original_image_list = []
    opticalflow_image_list = []
    label_list = []
    with open('val.txt', 'r') as f:
        for line in f:
            filename, label = line.split(' ')
            filename = '/'.join(filename.split('/')[-2:])
            original_image_list.append(os.path.join(original_data_dir,filename))
            opticalflow_image_list.append(os.path.join(opticalflow_data_dir,filename))
            label_list.append(int(label))
    return original_image_list, opticalflow_image_list, label_list

'''
trying to use the pipeline, but doesn't work for my custom case
'''
def input_pipeline(batch_size = 50):
    mean = np.load('ilsvrc_2012_mean.npy').astype('float32')
    mean = np.rollaxis(mean, 0, 3)
    #print(mean.shape, mean.dtype)
    mean_file = tf.constant(mean, name='mean_file')
    original_image_list, opticalflow_image_list, label_list = get_filenames()
    original_q = tf.train.string_input_producer(original_image_list)
    #print(type(original_q))
    opticalflow_q = tf.train.string_input_producer(opticalflow_image_list)
    labels = tf.constant(label_list)
    labels = tf.reshape(labels, [-1, 1])
    print(labels.get_shape())
    orig_reader = tf.WholeFileReader()
    #print(type(orig_reader))
    # read the whole image
    _, orig_value = orig_reader.read(original_q)
    #print(type(orig_value))
    #print(dir(orig_value))
    #print(orig_value.get_shape())
    # decode the image
    orig_img = tf.to_float(tf.image.decode_jpeg(orig_value)) - mean_file
    orig_img = tf.image.resize_image_with_crop_or_pad(orig_img, 227, 227)

    flow_reader = tf.WholeFileReader()
    # read the whole image
    _, flow_value = flow_reader.read(opticalflow_q)
    # decode the image and subtract from it the mean file
    flow_img = tf.to_float(tf.image.decode_jpeg(flow_value)) - mean_file
    flow_img = tf.image.resize_image_with_crop_or_pad(flow_img, 227, 227)

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 3 * batch_size
    original_batch, opticalflow_batch, label_batch = tf.train.batch(
        [orig_img, flow_img, labels], batch_size=batch_size)
    return original_batch, opticalflow_batch, label_batch

'''generator to get batch'''
class get_batch(object):
    def __init__(self, batch_size=50):
        original_data_dir = '/shared/original/thumbs'
        opticalflow_data_dir = '/shared/flow/thumbs'
        self.original_image_list = []
        self.opticalflow_image_list = []
        self.label_list = []
        with open('val.txt', 'r') as f:
            for line in f:
                filename, label = line.split(' ')
                filename = '/'.join(filename.split('/')[-2:])
                self.original_image_list.append(os.path.join(original_data_dir,filename))
                self.opticalflow_image_list.append(os.path.join(opticalflow_data_dir,filename))
                self.label_list.append(int(label))
        self.pointer = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.pointer < len(self.original_image_list):
            orig = self.original_image_list[self.pointer:self.pointer + self.batch_size]
            flow = self.opticalflow_image_list[self.pointer:self.pointer + self.batch_size]
            labels = self.label_list[self.pointer:self.pointer + self.batch_size]
            #read files
            def read_file(filename):
                img = misc.imread(filename)
                return misc.imresize(img, (227, 227, 3))
            '''a list of orig_imgs'''
            orig_imgs = [read_file(fname) for fname in orig]
            flow_imgs = [read_file(fname) for fname in flow]
            self.pointer += self.batch_size
            return (orig_imgs, flow_imgs, labels)
        else:
            raise StopIteration()

def main():
    with tf.Session() as sess:
        '''load caffenets'''
        orig, flow, comb, orig_in, flow_in = load_caffenets(sess)

        correct_num = [0,0,0]
        total_num = 0
        for batch in get_batch():
            orig_prob, flow_prob, comb_prob = sess.run([orig, flow, comb],
                    feed_dict={orig_in: batch[0], flow_in: batch[1]})
            total_num += len(batch[2])
            correct_num[0] += sum(np.argmax(orig_prob, 1) == batch[2])
            correct_num[1] += sum(np.argmax(flow_prob, 1) == batch[2])
            correct_num[2] += sum(np.argmax(comb_prob, 1) == batch[2])
            print('{}: cor {} {} {} total {}'.format(total_num, correct_num[0],
                correct_num[1], correct_num[2], total_num))
            #print orig_prob[0], flow_prob[0], comb_prob[0]
            #print np.argmax(orig_prob, 1), np.argmax(flow_prob, 1), np.argmax(comb_prob, 1), batch[2]
            #raw_input('press enter to continue...')

if __name__ == '__main__':
    main()
