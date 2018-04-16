import numpy as np
from scipy import misc
import scipy
import tensorflow as tf
import time
from PIL import Image
import os

FROZEN_FPATH = 'model/keras_vgg19_frozen_model.pb' # ADJUST
INPUTE_NODE = 'input_1:0' # ADJUST
OUTPUT_NODE = 'dense_2/Softmax:0' # ADJUST
CLASSES =["daisy", "dandelion", "roses", "sunflowers", "tulips"]
CROP_SIZE = (224, 224) # ADJUST
IMAGE_DIR = '/home/amax/john/tensorrt_demo2/val/roses'
BATCH_SIZE = 1

def image_to_np_CHW(image): 
    return np.asarray(
        image.resize(
            (224, 224), 
            Image.ANTIALIAS
        )).astype(np.float32)
#        )).transpose([2,0,1]).astype(np.float32)

def load_and_preprocess_images():
    file_list = [f for f in os.listdir(IMAGE_DIR) if os.path.isfile(os.path.join(IMAGE_DIR, f))]
    images_trt = []
    for f in file_list:
        images_trt.append(image_to_np_CHW(Image.open(os.path.join(IMAGE_DIR, f))))

    images_trt = np.stack(images_trt)

    num_batches = int(len(images_trt) / BATCH_SIZE)

    images_trt = np.reshape(images_trt[0:num_batches * BATCH_SIZE], [
        num_batches,
        BATCH_SIZE,
        images_trt.shape[1],
        images_trt.shape[2],
        images_trt.shape[3]
    ])

    return images_trt

graph_def = tf.GraphDef()
with tf.gfile.GFile(FROZEN_FPATH, "rb") as f:    
    graph_def.ParseFromString(f.read())
    
graph = tf.Graph()
with graph.as_default():
    net_inp, net_out = tf.import_graph_def(
        graph_def, return_elements=[INPUTE_NODE, OUTPUT_NODE])
    
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.Session(graph=graph, config=sess_config)

images_trt = load_and_preprocess_images()

print "Warm up..."
for image in images_trt:
    _ = sess.run(net_out, feed_dict={net_inp:image})


print "Start inferncing benchmark..."
results = []
correct_cnt = 0
process_time = []

for image in images_trt:
    start_t = time.time()
    result = sess.run(net_out, feed_dict={net_inp: image})
    process_time.append(time.time() - start_t)
    print "Inference time: {} ms".format(process_time[-1]*1000)
    print result
    result = CLASSES[np.argmax(result)]
    print result
    if result == 'roses':
       correct_cnt += 1
print "Accuracy is: {} ".format(correct_cnt/100.)
print "Mininum process time: {} ms".format(min(process_time)*1000)
print "Average process time: {} ms".format(1000 * sum(process_time)/len(process_time))
