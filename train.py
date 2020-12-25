import tensorflow as tf
from config import get_config
from model import Model
from base.base_train import Base_Train
import os
from dataloader import dataloader
import imageio
import numpy as np
from evalute import validation_model
os.environ['CUDA_VISIBLE_DEVICES']='4'

def main():
    is_training=True
    #get config
    config=get_config(is_training=is_training)

    #Instantiate train and model
    train=Base_Train(config)
    model=Model(config)

    #inference
    model.inference('net')

    #loss and optimizer
    loss=Base_Train.loss(tf.squeeze(model.logits, 3),tf.squeeze(model.annotation, 3))
    optimizer=Base_Train.optimizer('adam',loss,config.lr)

    #load post process op
    postprocess_module = tf.load_op_library(config.so_path)
    postprocess = postprocess_module.seg2_point_num(tf.cast(model.y*255, tf.uint8))
    postprocess = tf.identity(postprocess,name='net_output')

    #create session and load model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train.init_saver()
    train.load(sess)

    #load dataset
    train_records, valid_records = dataloader.get_datasetlist(config.data_dir)
    train_dataset_reader = dataloader(train_records,augument_flag=True)
    validation_dataset_reader = dataloader(valid_records,augument_flag=False)

    print('start training ......')
    for itr in range(1,config.max_iter):
        # load and feed data
        train_images, train_annotations, train_points = train_dataset_reader.next_batch(config.batch_size)
        feed_dict = {model.image: train_images, model.annotation: train_annotations}
        # run op
        _, train_loss, net_output, output = sess.run([optimizer, loss, model.y, postprocess], feed_dict=feed_dict)

        if itr % 10 == 0:
            print("Step: %d, Train_loss:%g : " % (itr, train_loss))
        if itr % 1000==0 :
            # save model
            train.saver.save(sess, config.weight_dir + "model.ckpt", itr)
            #debug save img
            imageio.imwrite(config.log_train_dir + str(itr) + '_org' + ".jpg",(train_images[0]).astype(np.uint8))
            imageio.imwrite(config.log_train_dir + str(itr) + '_out' + ".png",(255 * net_output[0]).astype(np.uint8))
            imageio.imwrite(config.log_train_dir + str(itr) + '_gt'  + ".png",(255 * train_annotations[0]).astype(np.uint8))
        if itr % 20000   == 0:
            #validation model
            validation_model(sess,valid_records, validation_dataset_reader, model.y, postprocess, model.image,config.log_validation_dir)




if __name__ == '__main__':
    main()
