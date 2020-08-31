import numpy as np
from pathlib import Path
import os
import json
import xml.etree.ElementTree as ET


def main():
    cur_dir = Path('.')

    # Raw data:
    xmls_dir = cur_dir/'psylla_raw/xml'
    imgs_dir = cur_dir/'psylla_raw/psy_index/'
    xmls = sorted(list(xmls_dir.glob('*.xml')))
    imgs = sorted(list(imgs_dir.glob('*.jpg')))

    # Create folders for processed data train/test/validation
    split = ['train', 'val', 'test']
    for split_name in split:
        Path.mkdir(cur_dir/'psylla_voc'/split_name /
                   'images', parents=True, exist_ok=True)
        Path.mkdir(cur_dir/'psylla_voc'/split_name /
                   'xmls', parents=True, exist_ok=True)

    voc_dir = cur_dir/'psylla_voc'

    # get a random split for train/validation/test
    ds_size = len(imgs)
    split_ratio = [0.7, 0.15, 0.15]
    split_size = [int(np.floor(ds_size*x)) for x in split_ratio]


    np.random.seed(321)
    rand_indices = np.random.permutation(ds_size)
    train_indices = rand_indices[:split_size[0]]
    val_indices = rand_indices[split_size[0]:split_size[0]+split_size[1]]
    test_indices = rand_indices[-split_size[2]:]

    # Extract the splits:
    train_imgs = np.array(imgs)[train_indices]
    val_imgs = np.array(imgs)[val_indices]
    test_imgs = np.array(imgs)[test_indices]

    train_xmls = np.array(xmls)[train_indices]
    val_xmls = np.array(xmls)[val_indices]
    test_xmls = np.array(xmls)[test_indices]

    ds_ = {'train': (train_imgs, train_xmls),
           'val': (val_imgs, val_xmls),
           'test': (test_imgs, test_xmls)}

    for split_idx, split_type in enumerate(ds_):
        img_split, xml_split = ds_[split_type]

        for (jpg_file, xml_file) in zip(img_split, xml_split):
            tree = ET.parse(xml_file)
            root = tree.getroot()  # 'doc' element

            if (root.find('labeled').text == 'true'):
                image_id = xml_file.name[-7:-4]
                size = root.find('size')

                bndboxs = []
                names = []
                items = root.find('outputs').find('object').findall('item')
                for item in items:
                    bndboxs.append(item.find('bndbox'))
                    names.append(item.find('name'))

                # create the file structure
                new_root = ET.Element('annotation')
                filename = ET.SubElement(new_root, 'filename')
                filename.text = f'{image_id}.jpg'
                new_root.append(size)
                for name, bndbox in zip(names, bndboxs):
                    # ! Temporary: we should consider wheter to keep this class
                    if (name.text == 'small_Psylla'):
                        name.text = 'psylla'
                    object_ = ET.SubElement(new_root, 'object')
                    object_.append(name)
                    object_.append(bndbox)


                xml_name = voc_dir/split_type/'xmls'/f'{image_id}.xml'
                tree_ = ET.ElementTree(new_root)
                tree_.write(str(xml_name))

                # Copy jpg
                src = str(jpg_file.absolute())
                dst = str((voc_dir/split_type/'images'/f'{image_id}.jpg').absolute())
                os.popen(f'cp {src} {dst}')


if __name__ == "__main__":
    main()
