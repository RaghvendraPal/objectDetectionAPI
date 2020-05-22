import pickle
import os
import base64
try:
    from PIL import Image
except ImportError:
    import Image
try:
    from StringIO import StringIO ## for Python 2
except ImportError:
    from io import StringIO, BytesIO
from keras.layers import Input
from modules.nn_base import nn_base, rpn_layer, classifier_layer
from keras.models import Model
from modules.image_resize import format_img, get_real_coordinates
from modules.rpn_to_roi import rpn_to_roi
from modules.non_max_suppression import non_max_suppression_fast
from keras import backend as K
import numpy as np
import time

from optparse import OptionParser
from matplotlib import pyplot as plt
from modules.config import Config
import tensorflow as tf
import cv2
from flask import *  

app = Flask(__name__)  
graph = None
model_rpn = None
file_name = None

config_output_filename = os.path.join('models', 'model_vgg_config.pickle')
print("CONFIG FILE NAME : ", config_output_filename)
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.model_path = "models/model_frcnn_vgg.hdf5"
# If the box classification value is less than this, we ignore this box
bbox_threshold = 0.8
class_mapping = C.class_mapping
class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}


@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':
        f = request.files['file']
        img_name = f.filename
        print(img_name)
        if img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):  
        
            print("File name : ", f )
            num_features = 512

            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, num_features)

            img_input = Input(shape=input_shape_img)
            roi_input = Input(shape=(C.num_rois, 4))
            feature_map_input = Input(shape=input_shape_features)
            # define the base network (VGG here, can be Resnet50, Inception, etc)
            shared_layers = nn_base(img_input, trainable=True)

            # define the RPN, built on the base layers
            num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
            rpn_layers = rpn_layer(shared_layers, num_anchors)

            classifier = classifier_layer(feature_map_input, roi_input, C.num_rois, nb_classes=len(C.class_mapping))
            # tf_config = some_custom_config
            # deep_grap = tf.Graph()
            # with deep_grap.as_default():
            #     ### graph definition here
            #     ### graph definition here

            #     init = tf.initialize_all_variables()
            model_rpn = Model(img_input, rpn_layers)
            print('Loading weights from {}'.format(C.model_path))
            model_rpn.load_weights(C.model_path, by_name=True)
            # with tf.Session(graph=deep_grap) as sess:
            #     sess.run(init)
            # sess = tf.Session(graph=deep_grap)
            # sess.run(init)
            # graph = tf.get_default_graph()
            # K.set_session(sess)
            model_classifier_only = Model([feature_map_input, roi_input], classifier)

            model_classifier = Model([feature_map_input, roi_input], classifier)


            print('Loading weights from {}'.format(C.model_path))
            model_rpn.load_weights(C.model_path, by_name=True)
            model_classifier.load_weights(C.model_path, by_name=True)

            model_rpn.compile(optimizer='sgd', loss='mse')
            model_classifier.compile(optimizer='sgd', loss='mse')

            # print(request)
            f.save("static/upload_image/"+f.filename)  
            print(img_name)
            st = time.time()
            # filepath = os.path.join(img_name)
            filepath = "static/upload_image/"+f.filename
            print("filepath : ",filepath)

            img = cv2.imread(filepath)
            # print(img)
            # print(img.shape)
            # plt.imshow(img)
            X, ratio = format_img(img, C)

            X = np.transpose(X, (0, 2, 3, 1))
            # print(X)

            # get output layer Y1, Y2 from the RPN and the feature maps F
            # Y1: y_rpn_cls
            # Y2: y_rpn_regr
            # global model_rpn
            # global session
            # with session.graph.as_default():
            #     K.set_session(session)
            # global model_rpn
            # # global sess
            # global graph
            # with graph.as_default():
                # K.set_session(sess)
            [Y1, Y2, F] = model_rpn.predict(X)


            # Get bboxes by applying NMS 
            # # R.shape = (300, 4)
            R = rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)

            # # convert from (x1,y1,x2,y2) to (x,y,w,h)
            R[:, 2] -= R[:, 0]
            R[:, 3] -= R[:, 1]

            # # # apply the spatial pyramid pooling to the proposed regions
            bboxes = {}
            probs = {}

            for jk in range(R.shape[0]//C.num_rois + 1):
                ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
                if ROIs.shape[1] == 0:
                    break

                if jk == R.shape[0]//C.num_rois:
                    #pad R
                    curr_shape = ROIs.shape
                    target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
                    ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                    ROIs_padded[:, :curr_shape[1], :] = ROIs
                    ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                    ROIs = ROIs_padded

                [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

                # Calculate bboxes coordinates on resized image
                for ii in range(P_cls.shape[1]):
                    # Ignore 'bg' class
                    if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                        continue

                    cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                    if cls_name not in bboxes:
                        bboxes[cls_name] = []
                        probs[cls_name] = []

                    (x, y, w, h) = ROIs[0, ii, :]

                    cls_num = np.argmax(P_cls[0, ii, :])
                    try:
                        (tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
                        tx /= C.classifier_regr_std[0]
                        ty /= C.classifier_regr_std[1]
                        tw /= C.classifier_regr_std[2]
                        th /= C.classifier_regr_std[3]
                        x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)
                    except:
                        pass
                    bboxes[cls_name].append([C.rpn_stride*x, C.rpn_stride*y, C.rpn_stride*(x+w), C.rpn_stride*(y+h)])
                    probs[cls_name].append(np.max(P_cls[0, ii, :]))

            all_dets = []

            for key in bboxes:
                bbox = np.array(bboxes[key])

                new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)
                for jk in range(new_boxes.shape[0]):
                    (x1, y1, x2, y2) = new_boxes[jk,:]

                    # Calculate real coordinates on original image
                    (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                    cv2.rectangle(img,(real_x1, real_y1), (real_x2, real_y2), (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),4)

                    textLabel = '{}: {}'.format(key,int(100*new_probs[jk]))
                    all_dets.append((key,100*new_probs[jk]))

                    (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
                    textOrg = (real_x1, real_y1-0)

                    cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 1)
                    cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
                    cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)

            print('Elapsed time = {}'.format(time.time() - st))
            print(all_dets)

            # plt.figure(figsize=(10,10))
            # plt.grid()
            global file_name
            file_name = "static/object_detect/"+f.filename
            print(file_name)
            cv2.imwrite(file_name, cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            print(type(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) 
            try:
                im = Image.open(file_name)
                # im.thumbnail((w, h), Image.ANTIALIAS)
                buffer = BytesIO()
                im.save(buffer, format='JPEG')
                print(buffer)
                encoded_img = base64.encodebytes(buffer.getvalue()).decode('ascii')
                # return Response(buffer, mimetype='image/jpeg')
                return jsonify({"status": 200, "image":encoded_img, "image_name":f.filename})

            except IOError:
                abort(404)
        else:
            print("File should be end with '.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff'")

        
if __name__ == '__main__':  
    app.run(debug = True)  
