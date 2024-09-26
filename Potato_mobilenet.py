import os
import random
import xml.etree.ElementTree as ET

import numpy as np

from mrcnn_mobilenet.config import Config
from mrcnn_mobilenet.utils import Dataset
from mrcnn_mobilenet.model import MaskRCNN
from mrcnn_mobilenet.model import load_image_gt
from mrcnn_mobilenet.model import mold_image
from mrcnn_mobilenet.utils import compute_ap

import warnings
warnings.filterwarnings("ignore")

Server = True
Server = False

Train = True
# Train = False

class KangarooDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):

        self.add_class("dataset", 1, "kangaroo")

        images = dataset_dir + '/images_init/'
        annots = dataset_dir + '/annots/'

        for filename in sorted(os.listdir(images)):
            img_id = filename[:-4]

            if img_id in ['00090']:
                continue

            if is_train and int(img_id) >= 150:
                continue

            if not is_train and int(img_id) < 150:
                continue

            img_path = images + filename
            ann_path = annots + img_id + '.xml'

            self.add_image('dataset', image_id=img_id, path=img_path, annotation=ann_path)

    def extract_bboxes(self, filename):

        tree = ET.parse(filename)
        root = tree.getroot()
        
        # extract each bounding box
        
        boxes = []
        
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        
        # extract image dimensions
        
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
    
        return boxes, width, height

    def load_mask(self, img_id):

        info = self.image_info[img_id]
        # define bbox coords location
        path = info['annotation']
        # load XML for file
        boxes, w, h = self.extract_bboxes(path)
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []

        for i in range(len(boxes)):
            box = boxes[i]
            row_min, row_max = box[1], box[3]
            col_min, col_max = box[0], box[2]

            masks[row_min:row_max, col_min:col_max, i] = 1
            class_ids.append(self.class_names.index('kangaroo'))

        return masks, np.asarray(class_ids, dtype='uint8')

    # load an image reference
    def image_reference(self, img_id):
        info = self.image_info[img_id]
        return info['path']

# get the images for training and testing names
def get_dataset_train_test_names(dataset_dir, percentage=0.8):
    images = dataset_dir + '/images/'
    annots = dataset_dir + '/annots/'

    img_names_list = [item[:-4] for item in os.listdir(images)]
    annots_names_list = [item[:-4] for item in os.listdir(annots)]

    img_names_list = list(filter(lambda x: x in annots_names_list, img_names_list))
    img_names_list = [item + '.npz' for item in img_names_list]

    # generate train and test set:
    random.shuffle(img_names_list)
    split_index = int(len(img_names_list) * percentage)

    train_set_names = img_names_list[:split_index]
    test_set_names = img_names_list[split_index:]

    return train_set_names, test_set_names , img_names_list

class DeepPotatoDataset(Dataset):

    def load_dataset(self, dataset_dir, dataset_names):
        # add class_ids
        for i in range(11):
            self.add_class("dataset", i, str(i))

        images = dataset_dir + "\images\\"
        annots = dataset_dir + "\\annots\\"

        for filename in sorted(dataset_names):

            img_id = filename[:-4]
            img_path = images + filename
            ann_path = annots + img_id + '.xml'
            tree = ET.parse(ann_path)
            root = tree.getroot()
            target = int(float(root.find('.//object/name').text))

            self.add_image('dataset', image_id=img_id, path=img_path, annotation=ann_path, class_name=target)

    def extract_bboxes(self, filename):

        tree = ET.parse(filename)
        root = tree.getroot()

        # extract each bounding box

        boxes = []

        for box in root.findall('.//bndbox'):
            xmin = int(float(box.find('xmin').text))
            ymin = int(float(box.find('ymin').text))
            xmax = int(float(box.find('xmax').text))
            ymax = int(float(box.find('ymax').text))
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)

        # extract image dimensions

        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)

        return boxes, width, height

    def load_mask(self, img_id):

        info = self.image_info[img_id]
        # define bbox coords location
        path = info['annotation']
        class_name = info['class_name']
        # load XML for file
        boxes, w, h = self.extract_bboxes(path)
        img_data = np.load(os.getcwd() + '\Deep Potato\images\\' + str(info['id']) + '.npz')
        h = img_data['mask'].shape[0]
        w = img_data['mask'].shape[1]
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = []

        for i in range(len(boxes)):
            masks[:, :, i] = img_data['mask'][:, :, 0]
            class_ids.append(self.class_names.index(str(class_name)))

        return masks, np.asarray(class_ids, dtype='uint8')

    def num_images(self):
        return len(self.image_info)

    # load an image reference
    def image_reference(self, img_id):
        info = self.image_info[img_id]
        return info['path']

def set_config_params(img_names_list):
    Config.IMAGE_CHANNEL_COUNT = 150

    mean_pixel_total = []
    Config.MEAN_PIXEL = []
    for img_n in img_names_list:
        img_ds = np.load(os.getcwd() + '\Deep Potato\images\\' + img_n)
        img = img_ds['data']
        mean_pixel_img = []
        for i in range(Config.IMAGE_CHANNEL_COUNT):
            img_i = img[:, :, i]
            mean_img_i = np.mean(img_i)
            mean_pixel_img.append(mean_img_i)
        mean_pixel_total.append(mean_pixel_img)
    for i in range(Config.IMAGE_CHANNEL_COUNT):
        mean_i_elements = [sub_list[i] for sub_list in mean_pixel_total]
        mean_img_i= np.mean(mean_i_elements)
        Config.MEAN_PIXEL.append(mean_img_i)



dataset_dir = os.getcwd() + '\Deep Potato'

train_set_names, test_set_names, img_names_list = get_dataset_train_test_names(dataset_dir, 0.7)


# training set
train_set = DeepPotatoDataset()
train_set.load_dataset(dataset_dir, train_set_names)
train_set_len = train_set.num_images()
train_set.prepare()

# testing set
test_set = DeepPotatoDataset()
test_set.load_dataset(dataset_dir, test_set_names)
test_set.prepare()

# change MEAN_PIXEL & NUMBER_CHANNLES params
set_config_params(img_names_list)

class PotatoPredictionConfig(Config):
    NAME = 'potato_config'
    # State number of classes inc. background
    NUM_CLASSES = 1 + 11 # ['BG', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    LEARNING_RATE = 0.01
    STEPS_PER_EPOCH = 1

# prepare config

config = PotatoPredictionConfig()

# set up the STEPS_PER_EPOCHS
config.STEPS_PER_EPOCH = (config.STEPS_PER_EPOCH * train_set_len) // (config.GPU_COUNT * config.IMAGES_PER_GPU)


config.display()

#define the model
model = MaskRCNN(mode='training', model_dir='./mrcnn_mobilenet/', config=config)

# load weights (mscoco)
if Train:
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=7, layers='heads')


if Server:
    weights = '/home/ubuntu/kangaroo_config20240729T0916/mask_rcnn_kangaroo_config_000.h5'
else:
    weights = './mrcnn_mobilenet/potato_config20240925T0503/mask_rcnn_potato_config_0007.h5' # we need to change this based on the generated model run

print("weights value: ", weights)
class PredictionConfig(Config):
    NAME = 'potato_config'
    # State number of classes inc. background
    NUM_CLASSES = 1 + 11
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = PredictionConfig()

if not Train:

    model = MaskRCNN(mode='inference', model_dir='./dist', config=config)

    model.load_weights(weights, by_name=True)

    def evaluate_model(dataset, model, cfg):
        APs = list()
        for image_id in dataset.image_ids:
            # load image, bounding boxes and masks for the image id
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)

            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = np.expand_dims(scaled_image, 0)
            cfg.BATCH_SIZE = 1
            # make prediction
            yhat = model.detect(sample, verbose=1)
            print("yhat: ", yhat)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP,precisions, recalls, overlaps = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            # store
            APs.append(AP)


        # calculate the mean AP across all images_init
        mAP = np.mean(APs)

        return mAP

    print('predicting on train set...')
    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, config)
    print("Train mAP: %.3f" % train_mAP)
    # evaluate model on validation dataset
    test_mAP = evaluate_model(test_set, model, config)
    print("Test mAP: %.3f" % test_mAP)
