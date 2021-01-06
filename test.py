import tensorflow as tf
from config import get_config
import os
from dataloader import dataloader
from evalute import validation_model
from evalute import evaluate_model,get_PR
import numpy as np
import cv2
import math
import imageio
from glob import glob
import time
os.environ['CUDA_VISIBLE_DEVICES']='4'

# 分类器阈值 大于阈值为阳性，小于阈值为阴性
classifier_threshold = 0.5
# 后处理连通域筛选面积倍率
areamulti_threshold = 0.3
# 后处理二值化阈值
binarization_threshold = 100

def main():
    is_training=False
    #get config
    config=get_config(is_training=is_training)

    #load dataset
    _,valid_records = dataloader.get_datasetlist(config.data_dir)
    validation_dataset_reader = dataloader(valid_records,augument_flag=False)

    if config.testmodel=="test1":
        #load post process op
        postprocess_module = tf.load_op_library(config.so_path)

        #create session and load model
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        with tf.gfile.FastGFile(config.pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input, output, net_output, info, get_info = tf.import_graph_def(graph_def,
                    return_elements=["input:0", "output:0","net_output:0", "info:0","get_info:0"])

        info_ = sess.run([info], feed_dict={get_info: 0})
        print(info_)

        validation_model(sess, valid_records, validation_dataset_reader, net_output, output, input,config.log_test_dir)

    elif config.testmodel=="test2":
        #create session and load model
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        with tf.gfile.FastGFile(config.pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input, output, info, get_info = tf.import_graph_def(graph_def,
                    return_elements=["input:0", "output:0", "info:0","get_info:0"])

        with tf.gfile.FastGFile(config.pb2_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input2, output2, = tf.import_graph_def(graph_def, return_elements=["input:0", "output:0"])

        TP, FP, FN = 0, 0, 0
        for itr in range(len(valid_records)):
            valid_images, valid_annotations, valid_points = validation_dataset_reader.next_batch(1)
            # run model 1
            output1_ = sess.run(output, feed_dict={input: valid_images})

            # img*255  from [0,1] to [0,255]
            post_in = (output1_ * 255).astype(np.uint8)

            # post process
            post_output = post_process(post_in)

            # get point rgb mean
            rgb_mean = get_rgbmean(post_output, valid_images)

            # run model2 to classifier
            rgbclassifier_output = sess.run(output2, feed_dict={input2: rgb_mean})

            # Get Positive or Negetive type
            final_output = post_output
            final_type = np.zeros([len(rgbclassifier_output), 1])
            for i in range(len(rgbclassifier_output)):
                if rgbclassifier_output[i] > classifier_threshold:
                    class_type = 1
                else:
                    class_type = 0
                final_type[i] = class_type

            # final_output+final_type  ->  [n,2]+[n,1]=[n,3]
            final_output = np.concatenate((final_output, final_type), axis=1)

            final_output = np.expand_dims(final_output, 0)
            TP_, FP_, FN_ = evaluate_model(valid_points, final_output)
            TP += TP_;FP += FP_;FN += FN_
            precious, recall, F1_Measure = get_PR(TP, FP, FN)
            print(' Precious: %.2f     Recall: %.2f    F1_Measure: %.2f  TP: %d  FP: %d  FN: %d ' % (
                precious * 100, recall * 100, F1_Measure * 100, TP, FP, FN))

            # debug draw result
            img_out = valid_images[0].astype(np.uint8)
            print('point num : ', len(post_output))
            for i in range(post_output.shape[0]):
                pointx1 = post_output[i][0].astype(np.uint)
                pointy1 = post_output[i][1].astype(np.uint)
                rgb_value = final_output[0][i][2]
                if rgb_value == 1:
                    rgbschar = (255, 0, 0)  # red
                else:
                    rgbschar = (0, 255, 0)  # green
                cv2.circle(img_out, (pointx1, pointy1), 3, rgbschar, -1)
            imageio.imwrite(config.log_test_dir + str(itr) +"_.png", img_out)

            image_co = valid_images[0].astype(np.uint8)
            img_out = cv2.cvtColor(image_co, cv2.COLOR_RGB2BGR)
            for i in range(len(valid_points[0])):
                pointx1 = valid_points[0][i][0]
                pointy1 = valid_points[0][i][1]
                cv2.circle(image_co, (pointx1, pointy1), 5, (255, 0, 0), 2)
            for i in range(post_output.shape[0]):
                pointx1 = post_output[i][0].astype(np.uint)
                pointy1 = post_output[i][1].astype(np.uint)
                cv2.circle(image_co, (pointx1, pointy1), 3, (0, 255, 0), -1)
            imageio.imwrite(config.log_test_dir + str(itr) + "_co.png", img_out)
            print("Saved image: %d" % itr)
    elif config.testmodel=="test3":
        #load post process op
        postprocess_module = tf.load_op_library(config.so_path)

        #create session and load model
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        with tf.gfile.FastGFile(config.pb_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input, output, net_output, info, get_info = tf.import_graph_def(graph_def,
                    return_elements=["input:0", "output:0","net_output:0", "info:0","get_info:0"])

        with tf.gfile.FastGFile(config.pb2_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            input2, output2, = tf.import_graph_def(graph_def, return_elements=["input:0", "output:0"])

        info_ = sess.run([info], feed_dict={get_info: 0})
        print(info_)
        print("testing ..........")
        img_list = glob(config.test_img_dir + '*.jpg')
        totle_time = 0
        for idx,img_path in enumerate(img_list):
            name = os.path.split(img_path)[1]
            img_name = os.path.splitext(name)[0]
            img = imageio.imread(img_path)
            img = np.expand_dims(img, 0)
            net_output_, output_ = sess.run([net_output, output], feed_dict={input: img})

            output_=np.squeeze(output_,0)
            # get point rgb mean
            rgb_mean = get_rgbmean(output_, img)
            # run model2 to classifier
            rgbclassifier_output = sess.run(output2, feed_dict={input2: rgb_mean})
            # Get Positive or Negetive type
            final_output = output_
            for i in range(len(rgbclassifier_output)):
                if rgbclassifier_output[i] > classifier_threshold:
                    final_output[i][2] = 1
                else:
                    final_output[i][2] = 0

            img_out = img[0].astype(np.uint8)
            for i in range(final_output.shape[0]):
                pointx1 = final_output[i][0].astype(np.uint)
                pointy1 = final_output[i][1].astype(np.uint)
                rgb_value = final_output[i][2]
                if rgb_value == 1:
                    rgbschar = (255, 0, 0)  # red
                else:
                    rgbschar = (0, 255, 0)  # green
                cv2.circle(img_out, (pointx1, pointy1), 3, rgbschar, -1)
            imageio.imwrite(config.log_test_dir +img_name + "_co.png", img_out)
            # imageio.imwrite(config.log_test_dir +img_name + "_mask.png", (255 * net_output_[0]).astype(np.uint8))
            print("processing ",idx," img: ",img_name)

# image pad
def image_pad(img):
    wid_ = img.shape[0]
    height_ = img.shape[1]
    wid = math.ceil(wid_ / 32.0) * 32
    wid = math.ceil(height_ / 32.0) * 32
    imgout = cv2.resize(img, (wid, wid))
    return imgout

# Post-processing algorithm
def post_process(img):
    average_area = 78.5

    # threshold
    _, img = cv2.threshold(img, binarization_threshold, 255, cv2.THRESH_BINARY)

    # Closed operation
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Connected domain algorithm
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, ltype=4)

    # select connected area  area_threshold*average_area
    stats_ = stats[stats[:, 4] > areamulti_threshold * average_area]

    # covert left up point to center point
    points_ = stats_[1:, 0:2]
    wid_height = stats_[1:, 2:4]
    post_out = points_ + wid_height // 2
    return post_out

# get point rgb mean
def get_rgbmean(post_output, img):
    # set 15*15 field
    roi_wh = 15
    roi_radius = roi_wh // 2

    # get 15*15 filed rgb mean
    for i, point in enumerate(post_output):

        # get point x y
        x = point[0]
        y = point[1]

        # Make sure not to cross the line +-roi_radius
        xtart = int(max(0, x - roi_radius))
        xend = int(min(img.shape[1], x + roi_radius + 1))
        ytart = int(max(0, y - roi_radius))
        yend = int(min(img.shape[2], y + roi_radius + 1))

        # get 15*15 filed
        roi = img[:, ytart:yend, xtart:xend, :]

        # get average rgb mean
        roi_mean_ = np.expand_dims(np.array([np.mean(roi[:, :, :, i]) for i in range(3)]), 0)
        if i == 0:
            rgb_mean = roi_mean_
        else:
            rgb_mean = np.concatenate((rgb_mean, roi_mean_), axis=0)
    if len(post_output)==0:
        rgb_mean=np.zeros((0,3))
    return rgb_mean




if __name__ == '__main__':
    main()
