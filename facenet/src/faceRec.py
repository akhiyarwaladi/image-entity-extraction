from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC 
from detect_faces import detect_face, detect_tiny_face
from keras.utils.data_utils import get_file
from wide_resnet import WideResNet
from keras import backend as K

sys.path.append('../gender-age')
import face_model
import datetime
import mxnet as mx



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help = 'Path of the video you want to test on.', default = 0)
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
    parser.add_argument('--model', default='../gender-age/model/model,0', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')

    args = parser.parse_args()
    
    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/Entity/Entity_margin_svm.pkl'
    VIDEO_PATH = args.path
    FACENET_MODEL_PATH = 'Models/facenet/20180402-114759.pb'
    
    
    # Load The Custom Classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    # # Load age and gender model
    # depth = 16
    # k = 8
    # margin = 0.4
    # weight_file = "Models/weights.28-3.73.hdf5"
    # model_age_gender = WideResNet(64, depth=depth, k=k)()
    # model_age_gender.load_weights(weight_file)     
    model_age_gender = face_model.FaceModel(args)   
            
    with tf.Graph().as_default():
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "./src/align")
            
            
            people_detected = set()
            person_detected = collections.Counter()

            
            frame = cv2.imread(VIDEO_PATH)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            # bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            # bounding_boxes = detect_tiny_face(frame)
            bounding_boxes = detect_face(frame)

            print(bounding_boxes)
            print(len(bounding_boxes))
            print(bounding_boxes.shape)
            faces_found = bounding_boxes.shape[0]

            entity = {}


            try:
                if faces_found > 0:
                    det = bounding_boxes[:, 0:4]
                    bb = np.zeros((faces_found, 4), dtype=np.int32)
                    unknown_faces = []
                    unknown_coor = []
                    for i in range(faces_found):
                        bb[i][0] = det[i][0]
                        bb[i][1] = det[i][1]
                        bb[i][2] = det[i][2]
                        bb[i][3] = det[i][3]

                        x, y = bb[i][0], bb[i][1]
                        w = bb[i][2] - x
                        h = bb[i][3] - y

                        x_pad_size = round( w * 0.01 )
                        y_pad_size = round( h * 0.01 )
                        x -= x_pad_size
                        y -= y_pad_size
                        w += 2*x_pad_size
                        h += 2*y_pad_size

                        bb[i][0] = x
                        bb[i][1] = y
                        bb[i][2] = x+w
                        bb[i][3] = y+h

                    
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        cropped_ = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2]]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        best_name = class_names[best_class_indices[0]]
                        print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                    
                        
                        color = (0, 0, 0)
                        
                        if best_class_probabilities > 0.75:

                            name = class_names[best_class_indices[0]]
                            

                            if name == "unknown":

                                name_entity = "{}_{}".format(name,i)
                                unknown_faces.append(cv2.resize(np.copy(cropped_), (64, 64), interpolation=cv2.INTER_CUBIC))
                                unknown_coor.append((bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                                # color = (255, 255, 255)

                                aligned = np.transpose(cropped_, (2,0,1))
                                input_blob = np.expand_dims(aligned, axis=0)
                                data = mx.nd.array(input_blob)
                                db = mx.io.DataBatch(data=(data,))
                               
                                gender, age =  model_age_gender.get_ga(db)
                                print(gender, age)
                                coor = (bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                                entity[coor] = "{}_{}".format("F" if gender == 0 else "M",age)
                            else:

                                name_entity = name
                                coor = (bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                                entity[coor] = name_entity
                                # color = (0, 0, 255)

                            # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), color, 2)
                            # text_x = bb[i][0]
                            # text_y = bb[i][3] + 20  

                            # cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            # 1, color, thickness=1, lineType=2)
                            # cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            # 1, color, thickness=1, lineType=2)
                            person_detected[best_name] += 1
                            

                        else:
                            unknown_faces.append(cv2.resize(np.copy(cropped_), (64, 64), interpolation=cv2.INTER_CUBIC))
                            unknown_coor.append((bb[i][0], bb[i][1], bb[i][2], bb[i][3]))
                            # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 255, 255), 2)
                            # text_x = bb[i][0]
                            # text_y = bb[i][3] + 20  
                            name = "unknown"
                            aligned = np.transpose(cropped_, (2,0,1))
                            input_blob = np.expand_dims(aligned, axis=0)
                            data = mx.nd.array(input_blob)
                            db = mx.io.DataBatch(data=(data,))

                            gender, age =  model_age_gender.get_ga(db)
                            print(gender, age)
                            coor = (bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                            entity[coor] = "{}_{}".format("F" if gender == 0 else "M",age)
                            # name_entity = "{}_{}".format(name,i)
                            # cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            # 1, (255, 255, 255), thickness=1, lineType=2)
                            # cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            # 1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                            # entity[name_entity] = [bb[i][1], bb[i][3], bb[i][0], bb[i][2]]


            except Exception as e:
                print(e)
                pass
            
          
    # unknown_faces_stack = np.stack(unknown_faces, axis=0)
    # print(unknown_faces_stack.shape)
    # print(type(unknown_faces_stack))
    # results = model_age_gender.predict(unknown_faces_stack)
    # print("hahah")
    # predicted_genders = results[0]
    # ages = np.arange(0, 101).reshape(101, 1)
    # predicted_ages = results[1].dot(ages).flatten()
    # print(predicted_genders)
    # print(predicted_ages)

    # for i in range(len(predicted_ages)):
    #     label = "{}_{}".format(int(predicted_ages[i]), "M" if predicted_genders[i][0] < 0.5 else "F")
    #     entity[unknown_coor[i]] = label


    print(entity)


    for key, value in entity.items():
        if value.split("_")[-1] == "M" or value.split("_")[-1] == "F":
            color = (255, 255, 255)
        else:
            color = (0, 0, 255)
            
        text_x = key[0]  
        text_y = key[3] + 20 
        cv2.rectangle(frame, (key[0], key[1]), (key[2], key[3]), color, 2)
        cv2.putText(frame, value, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
        1, (255, 255, 255), thickness=1, lineType=2)

    cv2.imshow('Face Recognition',frame)
    cv2.waitKey(0)

            
            
if __name__ == '__main__':
    main()
