import json
import xml.etree.ElementTree as ET
import os
import pandas as pd
import numpy as np

def create_target_value(img_ind , RWC_val):
    df = pd.read_csv('./measurements/CHESS_FOLDS.CSV')
    NDVI_val = df[df['Sample_Index'] == img_ind]['NDVI'].values[0]
    target_val = (NDVI_val * 100 + RWC_val) / 2
    return target_val // 10

def load_img(img_ind):
    # the image has 2 keys : data and mask
    img = np.load('./images_init/' + str(img_ind) + '.npz')
    print("img keys: ", list(img.keys()))
    print("img mask : ", img['mask'])
    print("img data shape : ", img['data'].shape) # (nb_hyp_chan, y, x)

def access_geojson_file(geojson_path):
    with open(geojson_path) as f:
        data = json.load(f)
    return data

# Polygon_Coord: a List of the Polygon Coord for an img
def create_bndbox_coord(Polygon_Coord):
    x_coord = [xy_coord[0] for xy_coord in Polygon_Coord]
    y_coord = [xy_coord[1] for xy_coord in Polygon_Coord]

    x_coord_max = max(x_coord)
    x_coord_min = min(x_coord)
    y_coord_max = max(y_coord)
    y_coord_min = min(y_coord)

    return [x_coord_max, x_coord_min, y_coord_max, y_coord_min]

def create_size_xml(annotation, img_ind):

    # read the npz image:
    img = np.load('./images/'+str(img_ind)+'.npz')

    x_img = img['data'].shape[0]
    y_img = img['data'].shape[1]
    depth_img = img['data'].shape[2]

    size = ET.SubElement(annotation, "size")

    width = ET.SubElement(size, "width")
    width.text = str(y_img)

    height = ET.SubElement(size, "height")
    height.text = str(x_img)

    depth = ET.SubElement(size, "depth")
    depth.text = str(depth_img)

    return [x_img, y_img, depth_img]

def create_bndbox_obj_xml(annotation, Polygon_Coord, target_val, img_shape):

    coord = Polygon_Coord

    object = ET.SubElement(annotation, "object")

    name = ET.SubElement(object, "name")
    # name.text = "the Name of the class " # change this by the target value of the bndbox
    name.text = str(target_val)

    pose = ET.SubElement(object, "pose")
    pose.text = "Unspecified"

    truncated = ET.SubElement(object, "truncated")
    if(coord[1] < 0) or (coord[3] < 0) or (coord[0] > img_shape[1]) or (coord[2] > img_shape[0]):
        truncated.text = str(1)
    else:
        truncated.text = str(0)

    difficult = ET.SubElement(object, "difficult")
    difficult.text = str(0) # is not difficult to detect

    bndbox = ET.SubElement(object, "bndbox")

    xmin = ET.SubElement(bndbox, "xmin")
    xmin.text = str(coord[1])

    ymin = ET.SubElement(bndbox, "ymin")
    ymin.text = str(coord[3])

    xmax = ET.SubElement(bndbox, "xmax")
    xmax.text = str(coord[0])

    ymax = ET.SubElement(bndbox, "ymax")
    ymax.text = str(coord[2])

def create_xml_file(img_ind, data):
    # load img data
    img_data = np.load('./images/' + str(img_ind) + '.npz')
    img_mask = img_data['mask']
    new_mask = np.expand_dims(img_mask, axis=-1)

    # calculate mask indexes:
    true_ind = np.argwhere(new_mask == True)
    filtered_list = [item for item in true_ind if item[2] == 0]

    x_min = min(item[0] for item in filtered_list)
    x_max = max(item[0] for item in filtered_list)
    y_min = min(item[1] for item in filtered_list)
    y_max = max(item[1] for item in filtered_list)



    filter_x_y_min_min= (x_min , y_min)
    filter_x_y_max_min= (x_max , y_min)
    filter_x_y_min_max= (x_min , y_max)
    filter_x_y_max_max= (x_max , y_max)

    Polygon_Coord = [x_max, x_min, y_max, y_min]

    xml_file_path = './annots/'+str(img_ind)+'.xml'
    annotation = ET.Element("annotation")

    folder = ET.SubElement(annotation, "folder")
    folder.text = "Deep Potato"

    filename = ET.SubElement(annotation, "filename")
    filename.text = str(img_ind)+".npz"

    path = ET.SubElement(annotation, "path")
    path.text = r"C:\Users\raoua\Desktop\kangaroo-detection-mask-rcnn\Mask_RCNN\Deep Potato\images\A-potato.npz" # r indicates it's a raw string

    source = ET.SubElement(annotation, "source")

    database = ET.SubElement(source, "database")
    database.text = "Unknown"

    img_shape = create_size_xml(annotation, img_ind)

    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = str(0)

    # in case of multiple bndboxes in one img :
    for data_i in data:
        Polygon_Coord = [x_max, x_min, y_max, y_min]
        RWC_val = data_i['properties']['RWC']
        target_val = create_target_value(img_ind, RWC_val)
        create_bndbox_obj_xml(annotation, Polygon_Coord, target_val, img_shape)

    tree = ET.ElementTree(annotation)

    with open(xml_file_path, 'wb') as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)

    return "the xml file : " + xml_file_path + "is created"

def create_annot_files():
    # create the annots directory if it doesn't exist
    if not os.path.exists("./annots"):
        os.makedirs("./annots")

    data1 = access_geojson_file("./measurements/table1.geojson")
    sample_indices = [feature['properties']['Sample_Index'] for feature in data1['features']]
    imgs_ind = set(sample_indices)

    for img_ind in imgs_ind:
        data_img = [feature for feature in data1['features'] if feature['properties']['Sample_Index'] == img_ind]
        create_xml_file(img_ind, data_img)

    print("after generating the xml files")

# we should reformat the image and the mask
def reformat_images():
    os.makedirs("images")
    images_init = os.listdir('./images_init/')
    for filename in images_init:
        img_ind = filename[:-4]
        img_data = np.load('./images_init/' + str(img_ind) + '.npz')
        data_transposed = np.transpose(img_data['data'], (2, 1, 0))
        mask_transposed = np.transpose(img_data['mask'], (2, 1, 0))

        # save the formatted image to a new directory
        np.savez('./images/' + str(img_ind) + '.npz', data=data_transposed, mask=mask_transposed)

def print_mask_value(img_ind):
    img_data = np.load('./images/' + str(img_ind) + '.npz')
    print("img shape: ", img_data['data'].shape)
    print('mask shape: ', img_data['mask'].shape)
    img_mask = img_data['mask']
    # calculate mask indexes:
    true_ind = np.argwhere(img_mask == True)
    filtered_list = [item for item in true_ind if item[2] == 0]



create_annot_files()
# reformat_images()

# test
# annotation = ET.Element("annotation")
# create_size_xml(annotation, 1)
# create_target_value(47, 100)

# print_mask_value(2)