import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

import mrcnn
from mrcnn.config import Config
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.utils import Dataset
from mrcnn.model import MaskRCNN
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image
from mrcnn.utils import compute_ap

import warnings
warnings.filterwarnings("ignore")

Server = True
Server = False

# Train = True
Train = False

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

# get the imges for training and testing names
def get_dataset_train_test_names(dataset_dir, percentage=0.8):
    images = dataset_dir + '/images/'
    annots = dataset_dir + '/annots/'

    img_names_list = [item[:-4] for item in os.listdir(images)]
    annots_names_list = [item[:-4] for item in os.listdir(annots)]

    # img_names_list = annots_names_list & annots_names_list
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
model = MaskRCNN(mode='training', model_dir='./', config=config)


if Train:
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=7, layers='heads')

if Server:

    weights = '/home/ubuntu/mask_rcnn_potato_config20240729T0916/mask_rcnn_kangaroo_config_000.h5'
else:
    weights = './potato_config20240729T0916/mask_rcnn_potato_config_0007.h5'


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

    #visualize the architecture:
    #model.keras_model.summary()

    model.load_weights(weights, by_name=True)


    def evaluate_model(dataset, model, cfg):
        # model.keras_model.summary()
        APs = list()

        for image_id in dataset.image_ids:
            # load image, bounding boxes and masks for the image id
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id,
                                                                             use_mini_mask=False)
            # convert pixel values (e.g. center)
            scaled_image = mold_image(image, cfg)
            # convert image into one sample
            sample = np.expand_dims(scaled_image, 0)
            # make prediction
            yhat = model.detect(sample, verbose=1)
            # extract results for first sample
            r = yhat[0]
            # calculate statistics, including AP
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])

            # store
            APs.append(AP)

            return APs

    print('predicting on train set...')
    # evaluate model on training dataset
    train_mAP = evaluate_model(train_set, model, config)
    print("Train mAP:", train_mAP)

    test_mAP = evaluate_model(test_set, model, config)
    print("Test mAP:" ,test_mAP)

    def evaluate_image(dataset, model, cfg):

        APs = list()
        image_id = dataset.image_ids[0]
        print("image_id: ", dataset.image_ids)

        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id,
                                                                         use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=1)
        # extract results for first sample
        r = yhat[0]
        print("classification:" ,r["class_ids"])
        def transform_image_rgb(img,frames,classes):
            red_channel = img[:, :, 40]  # Corresponds to ~650 nm
            green_channel = img[:, :, 30]  # Corresponds to ~550 nm
            blue_channel = img[:, :, 20]  # Corresponds to ~450 nm

            # Stack the selected channels to form an RGB image
            rgb_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)

            # Normalize the image to 0-1 range for display
            rgb_image_normalized = (rgb_image - rgb_image.min()) / (rgb_image.max() - rgb_image.min())


            # Display the RGB image
            fig, ax = plt.subplots(1)
            plt.imshow(rgb_image_normalized)

            # Loop over the frames and add square frames (bounding boxes)
            for i, frame in enumerate(frames):
                x_min, y_min, x_max, y_max = frame
                # Create a Rectangle patch
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r',
                                         facecolor='none')
                # Add the rectangle to the plot
                ax.add_patch(rect)

                ax.text(x_min, y_min - 10, classes[i], color='black', fontsize=12, backgroundcolor='white')

            plt.title("RGB Visualization of Hyperspectral Image with rois")
            plt.show()

        def visualize_mask(image, rois_id, masks):
            # Create an RGB image by combining channels (e.g., using the first three channels)
            rgb_image = np.zeros((image.shape[0], image.shape[1], 3))
            rgb_image[:, :, 0] = np.mean(image[:, :, :50], axis=2)  # Red channel
            rgb_image[:, :, 1] = np.mean(image[:, :, 50:100], axis=2)  # Green channel
            rgb_image[:, :, 2] = np.mean(image[:, :, 100:], axis=2)  # Blue channel

            mask = masks[:, :, rois_id]
            mask_color = np.zeros_like(rgb_image)  # Create an empty RGB image

            # Colorize the mask (e.g., with a semi-transparent red)
            mask_color[:, :, 0] = mask * 255  # Red channel
            mask_color[:, :, 1] = 0  # Green channel
            mask_color[:, :, 2] = 0  # Blue channel

            # Blend the mask with the image
            alpha = 0.5  # Transparency factor
            blended_image = rgb_image * (1 - alpha) + mask_color / 255.0 * alpha

            # Plot the original image and the mask overlay
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(rgb_image)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            ax[1].imshow(blended_image)
            ax[1].axis('off')

            plt.show()

        # transform_image_rgb(image, r['rois'] , r["class_ids"])
        # visualize_mask(image,0 , r['masks'])


        # calculate the mean AP across all images_init
        # mAP = np.mean(APs)
        #
        # print('Mean Average Precision: {}'.format(mAP))
        # return mAP


   # evaluate_image(test_set, model, config)




    n_image = 1

    # for i in range(n_image):
    #     ids = test_set.image_ids
    #     print("ids: ", ids)
    #     print("image reference: ", test_set.image_reference(i))
    #     image = test_set.load_image(i)
    #     mask, _ = test_set.load_mask(i)
    #     scaled_image = mold_image(image, config)
    #     # convert image into one sample
    #     sample = np.expand_dims(scaled_image, 0)
    #     # make prediction
    #     print('detecting...')
    #     yhat = model.detect(sample, verbose=0)
    #     print("yhat: ", yhat)
    #     print("yhat mask shape value is: ", yhat[0]['masks'].shape)
    #     # # stage subplot
    #     # plt.subplot(n_image, 2, i*2+1)
    #     # plt.axis('off')
    #     # plt.imshow(image)
    #     # plt.title('Actual')
    #     # # plot masks
    #     # for j in range(mask.shape[2]):
    #     #     plt.imshow(mask[:,:,j], cmap='Blues', alpha=0.3)
    #     #
    #     # plt.subplot(n_image, 2, i*2+2)
    #     # plt.axis('off')
    #     # plt.imshow(image)
    #     # plt.title('Predicted')
    #     # ax = plt.gca()
    #     # # plot predicted masks
    #     # print("yhat: " , yhat)
    #     # yhat = yhat[0]
    #     # for box in yhat['rois']:
    #     #     # get coordinates
    #     #     y1, x1, y2, x2 = box
    #     #     width, height = x2 - x1, y2 - y1
    #     #     rect = Rectangle((x1, y1), width, height, fill=False, color='red')
    #     #     ax.add_patch(rect)

# plt.show()






#     # weights = '/home/david/Projects/strath/kangaroo/models/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'
# class PredictionConfig(Config):
#     NAME = 'kangaroo_config'
#     # State number of classes inc. background
#     NUM_CLASSES = 1 + 1
#     # simplify GPU config
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#
# if Server:
#
#     weights = '/home/ubuntu/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'
# else:
#     weights = './kangaroo_config20240722T0931/mask_rcnn_kangaroo_config_0001.h5'
#     # weights = './kangaroo_config20240717T1659/mask_rcnn_kangaroo_config_0002.h5'
#     # weights = '/home/david/Projects/strath/kangaroo/models/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'
#
# tst_weights = './kangaroo_config20240722T0931/mask_rcnn_kangaroo_config_0001.h5'
# # tst_weights = './kangaroo_config20240717T1659/mask_rcnn_kangaroo_config_0002.h5'
# # tst_weights = '/home/david/Projects/strath/data/kangaroo_config20191114T1607/mask_rcnn_kangaroo_config_0002.h5'
#
# # create config
# cfg = PredictionConfig()
# print("train value after training: ", Train)
# if not Train:
#
#     model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
#
#     model.load_weights(weights, by_name=True)
#
#     def evaluate_model(dataset, model, cfg):
#         APs = list()
#         for image_id in dataset.image_ids:
#             # load image, bounding boxes and masks for the image id
#             image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
#             # convert pixel values (e.g. center)
#             scaled_image = mold_image(image, cfg)
#             # convert image into one sample
#             sample = np.expand_dims(scaled_image, 0)
#             # make prediction
#             yhat = model.detect(sample, verbose=1)
#             # extract results for first sample
#             r = yhat[0]
#             # calculate statistics, including AP
#             AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
#             # store
#             APs.append(AP)
#
#         # calculate the mean AP across all images_init
#         mAP = np.mean(APs)
#
#         print('Mean Average Precision: {}'.format(mAP))
#         return mAP
#
#     print('predicting on train set...')
#     # # # evaluate model on training dataset
#     # train_mAP = evaluate_model(train_set, model, cfg)
#     # print("Train mAP: %.3f" % train_mAP)
#
#     # test_mAP = evaluate_model(test_set, model, cfg)
#     # print("Train mAP: %.3f" % test_mAP)
#
# n_image = 2
#
# for i in range(n_image):
#     image = test_set.load_image(i)
#     mask, _ = test_set.load_mask(i)
#     scaled_image = mold_image(image, cfg)
#     # convert image into one sample
#     sample = np.expand_dims(scaled_image, 0)
#     # make prediction
#     print('detecting...')
#     yhat = model.detect(sample, verbose=0)
#     # stage subplot
#     plt.subplot(n_image, 2, i*2+1)
#     plt.axis('off')
#     plt.imshow(image)
#     plt.title('Actual')
#     # plot masks
#     for j in range(mask.shape[2]):
#         plt.imshow(mask[:,:,j], cmap='Blues', alpha=0.3)
#
#     plt.subplot(n_image, 2, i*2+2)
#     plt.axis('off')
#     plt.imshow(image)
#     plt.title('Predicted')
#     ax = plt.gca()
#     # plot predicted masks
#     print("yhat: " , yhat)
#     yhat = yhat[0]
#     for box in yhat['rois']:
#         # get coordinates
#         y1, x1, y2, x2 = box
#         width, height = x2 - x1, y2 - y1
#         rect = Rectangle((x1, y1), width, height, fill=False, color='red')
#         ax.add_patch(rect)
#
# plt.show()