import imageio
import numpy as np
import math
import time
import cv2

def evaluate_model(labpoint,prepoint):
    MinDistance_Shreshold=6
    batch_size=labpoint.shape[0]
    TP = np.zeros([batch_size], np.int32)
    FP = np.zeros([batch_size], np.int32)
    FN = np.zeros([batch_size], np.int32)

    for i in range(batch_size):
        pre_len = len(np.nonzero(prepoint[i, :, 2])[0])
        # pre_len = prepoint.shape[1]
        for j in range(len(labpoint[i])):
            x1=labpoint[i][j][0]
            y1=labpoint[i][j][1]
            min_dist=MinDistance_Shreshold+0.00001
            for j in range(pre_len):
                x2=prepoint[i][j][0]
                y2=prepoint[i][j][1]
                square=(x1-x2)**2+(y1-y2)**2
                distance=math.sqrt(square)
                if distance<min_dist:
                    min_dist=distance
            if min_dist<=MinDistance_Shreshold:
                TP[i]+=1
            else:
                FN[i]+=1
        FP[i]=max(pre_len-TP[i],0)
    return np.sum(TP),np.sum(FP),np.sum(FN)


def get_PR(TP,FP,FN):
    try :
        precious = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_Measure = 2 * precious * recall / (precious + recall)
    except ZeroDivisionError:
        print("ZeroDivisionError")
        precious, recall, F1_Measure=0,0,0
    return precious,recall,F1_Measure

def validation_model(sess,valid_records,validation_dataset_reader,net_output,output,image,write_path):
    print("validation ..........")
    TP,FP,FN,totle_time= 0, 0 ,0,0
    for itr in range(len(valid_records)):
        valid_images, valid_annotations, valid_points = validation_dataset_reader.next_batch(1)
        time_start = time.time()
        net_output_, output_ = sess.run([net_output, output], feed_dict={image: valid_images})
        time_end = time.time()
        use_time = time_end - time_start
        if itr != 0:
            totle_time += use_time
        TP_, FP_, FN_=evaluate_model(valid_points, output_)
        output_=np.squeeze(output_,0)
        TP+=TP_ ; FP+=FP_ ; FN+=FN_
        print('TP_ %d    FP_ %d  FN_  %d' % (TP_ , FP_, FN_))
        precious, recall, F1_Measure = get_PR(TP, FP, FN)
        print('use time %.4f Precious: %.2f     Recall: %.2f    F1_Measure: %.2f' % (use_time,precious * 100, recall * 100, F1_Measure * 100))

        # """
        # debug
        image_co = valid_images[0].astype(np.uint8)
        for j in range(len(valid_points[0])):
            pointx1 = valid_points[0][j][0]
            pointy1 = valid_points[0][j][1]
            cv2.circle(image_co, (pointx1, pointy1), 5, (255, 0, 0), 2)
        for j in range(output_.shape[0]):
            pointx1 = output_[j][0].astype(np.uint)
            pointy1 = output_[j][1].astype(np.uint)
            cv2.circle(image_co, (pointx1, pointy1), 3, (0, 255, 0), -1)
        imageio.imwrite(write_path + str(itr) + "_co.png", image_co)
        # imageio.imwrite(write_path + str(itr) + '_org' + ".jpg", (valid_images[0]).astype(np.uint8))
        imageio.imwrite(write_path + str(itr) + '_out' + ".png", (255 * net_output_[0]).astype(np.uint8))
        imageio.imwrite(write_path + str(itr) + '_gt'  + ".png",(255 * valid_annotations[0]).astype(np.uint8))
        # """
    avguse_time = totle_time / (len(valid_records))
    precious, recall, F1_Measure = get_PR(TP, FP, FN)
    print('average use time: %.4f Precious: %.2f     Recall: %.2f    F1_Measure: %.2f' % (avguse_time,precious * 100, recall * 100, F1_Measure * 100))
    print("saving model - Step: %d," % (itr))