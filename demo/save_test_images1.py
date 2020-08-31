from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def extract_xml_bbox(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()  # 'doc' element

    glycaspis_bbox = []
    psyllaephagus_bbox = []
    objects = root.findall('object')
    for obj in objects:
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        xmax = int(bbox.find('xmax').text)
        ymin = int(bbox.find('ymin').text)
        ymax = int(bbox.find('ymax').text)
        p = 1
        if (obj.find('name').text == 'Psyllaephagus'):
            psyllaephagus_bbox.append(np.array([xmin, ymin, xmax, ymax, p]))
        else:
            glycaspis_bbox.append(np.array([xmin, ymin, xmax, ymax, p]))

    if (len(psyllaephagus_bbox) == 0):
        psylla_bbox = np.zeros((0, 5))
    else:
        psylla_bbox = np.array(psylla_bbox)

    if (len(glycaspis_bbox) == 0):
        wasp_bbox = np.zeros((0, 5))
    else:
        wasp_bbox = np.array(wasp_bbox)

    return [psylla_bbox, wasp_bbox]


def plot_gt(img, bboxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
    psyllaephagus_bbox, glycaspis_bbox = bboxes

    if psyllaephagus_bbox is not None:
        for bbox in psyllaephagus_bbox:
            x1, y1, x2, y2, p = [int(b) for b in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 20, 200), 2)
            cv2.putText(img, 'Psyllaephagus | GT', (x2, y2), font,
                        0.8, (200, 20, 200), 2, cv2.LINE_AA)

    if glycaspis_bbox is not None:
        for bbox in glycaspis_bbox:
            x1, y1, x2, y2, p = [int(b) for b in bbox]
            cv2.rectangle(img, (x1, y1), (x2, y2), (200, 20, 200), 2)
            cv2.putText(img, 'Glycaspis | GT', (x2, y2), font,
                        0.8, (200, 20, 200), 2, cv2.LINE_AA)

    return img


config_file = '/home/alex/Documents/GitHub/FrotiersInEcology/configs/faster_rcnn_r50_fpn_1x_psylla2.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/alex/Documents/GitHub/FrotiersInEcology/work_dirs/faster_rcnn_r50_fpn_1x_hazeka/epoch_100.pth'


model = init_detector(config_file, checkpoint_file, device='cuda:0')


imgs_dir = Path('/home/alex/Documents/GitHub/FrotiersInEcology/data/psylla_voc/test/images/')
xmls_dir = Path('/home/alex/Documents/GitHub/FrotiersInEcology/data/psylla_voc/test/xmls/')

xmls = sorted(list(xmls_dir.glob('*.xml')))
imgs = sorted(list(imgs_dir.glob('*.jpg')))

print('Starting to plot.')

plt.figure(figsize=(10, 10))

for img_path, xml_path in zip(imgs, xmls):
    print(f'Processing {img_path} and {xml_path}.')
    prediction = inference_detector(model, str(img_path))
    gt = extract_xml_bbox(xml_path)

#    img = show_result_pyplot(str(img_path), prediction,
#                             model.CLASSES, retimg=True)
    img = show_result_pyplot( model, str(img_path), prediction, retimg=True)#retimg= returned image from show_result_pyplot
    img = plot_gt(img, gt)
    plt.imshow((mmcv.bgr2rgb(img)))
    plt.savefig(Path('.')/'test_imgs'/img_path.name)
    #plt.savefig('/home/alex/mmdetection/data/psylla_voc/test/test_imgs'/img_path.name)

print('Done plotting.')
