from tensorrt.lite import Engine

from PIL import Image
import numpy as np
import os
import functools
import time

PLAN_single = '/home/amax/john/tensorrt_demo2/model/keras_vgg19_b1_fp32.engine'  # engine filename for batch size 1
PLAN_half = '/home/amax/john/tensorrt_demo2/model/keras_vgg19_b1_fp16.engine'
IMAGE_DIR = '/home/amax/john/tensorrt_demo2/val/roses'
BATCH_SIZE = 1


def analyze(output_data):
    LABELS=["daisy", "dandelion", "roses", "sunflowers", "tulips"]
    output = output_data.reshape(-1, len(LABELS))
    
    top_classes = [LABELS[idx] for idx in np.argmax(output, axis=1)]
    top_classes_prob = np.amax(output, axis=1)  

    return top_classes, top_classes_prob


def image_to_np_CHW(image): 
    return np.asarray(
        image.resize(
            (224, 224), 
            Image.ANTIALIAS
        )).transpose([2,0,1]).astype(np.float32)


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


def timeit(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs
    return newfunc

def load_TRT_engine(plan):
    engine = Engine(PLAN=plan, postprocessors={"dense_2/Softmax":analyze})   
    return engine

engine_single = load_TRT_engine(PLAN_single)
engine_half = load_TRT_engine(PLAN_half)


images_trt = load_and_preprocess_images()

# @timeit
def infer_all_images_trt(engine):
    results = []
    correct_cnt = 0
    process_time = []
    for image in images_trt:
        start_t = time.time()  
        result = engine.infer(image)
        process_time.append(time.time() - start_t)
        print "Inference time: {} ms".format(process_time[-1]*1000)
        #results.append(result)
        print result[0][0][0]
	if result[0][0][0] == 'roses':
           correct_cnt += 1
    print "Accuracy is: {} ".format(correct_cnt/100.)
    print "Mininum process time: {} ms".format(min(process_time)*1000)
    print "Average process time: {} ms".format(1000 * sum(process_time)/len(process_time))
    return results

# DO inference with TRT
print "Warm up..."
for image in images_trt:
    _ = engine_single.infer(image)

print "Single Presicion..."
infer_all_images_trt(engine_single)
print "Half precision..."
infer_all_images_trt(engine_half)


