import cv2
import numpy as np
import os
import re
from glob import glob

label_dict = {"built-up": (0, [255, 0, 0]),
              "farmland": (1, [0, 255, 0]),
              "forest": (2, [0, 255, 255]),
              "meadow": (3, [255, 255, 0]),
              "water": (4, [0, 0, 255]),
              "unknown": (5, [0, 0, 0])}

nclass = 5

img_dir = "raw_data/train/img_RGB"
label_dir = "raw_data/train/label_5classes"


def backbone_dataGen(img_dir, label_dir, dataset_dir="dataset/train", num_sample=2000):
    scale_L1 = 56
    scale_L2 = 112
    scale_L3 = 224

    l1 = int(scale_L1 / 2)
    l2 = int(scale_L2 / 2)
    l3 = int(scale_L3 / 2)
    scale_list = [l1, l2, l3]

    sampleFlag = np.array([0, 0, 0, 0, 0])
    threshold = 0.8

    img_list = os.listdir(img_dir)

    while np.any(sampleFlag<num_sample):
        l = scale_list[np.random.randint(0, 3)]   # sample a scale randomly
        img_name = img_list[np.random.randint(0, len(img_list))]  # sample a GF-2 img randomly
        label_name = img_name.split(".tif")[0] + "_label.tif"
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        label = cv2.cvtColor(cv2.imread(os.path.join(label_dir, label_name)), cv2.COLOR_BGR2RGB)

        width = img.shape[0]
        height = img.shape[1]

        img_padded = cv2.copyMakeBorder(img, l, l, l, l, cv2.BORDER_CONSTANT, value=label_dict["unknown"][1])
        label_padded = cv2.copyMakeBorder(label, l, l, l, l, cv2.BORDER_CONSTANT, value=label_dict["unknown"][1])

        center_x = np.random.randint(0, width)
        center_y = np.random.randint(0, height)

        img_cropped = img_padded[center_x - l:center_x + l, center_y - l:center_y + l, :]
        label_cropped = label_padded[center_x - l:center_x + l, center_y - l:center_y + l, :]

        label_flatten = label_cropped.reshape(-1, 3)
        label_sum = np.zeros(nclass + 1)
        for k, v in label_dict.items():
            label_sum[v[0]] = np.sum((label_flatten == v[1]).all(1))

        label_sum = label_sum / np.sum(label_sum)

        cid = np.argmax(label_sum)

        if cid < nclass and label_sum[cid] > threshold and sampleFlag[cid] < num_sample:
            sampleFlag[cid] += 1

            if not os.path.exists(os.path.join(dataset_dir, str(cid))):
                os.mkdir(os.path.join(dataset_dir, str(cid)))

            cv2.imwrite(os.path.join(dataset_dir, str(cid), str(cid) + "-" + str(sampleFlag[cid]) + ".tiff"),
                        cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))


def seg_dataGen(img_dir, label_dir, dataset_dir="SegDataset/train", target_size=224):
    img_names = [filename for _, _, filenames in os.walk(img_dir) for filename in filenames if filename.endswith(".tif")]
    num_save = 0
    target_height, target_width = target_size

    for img_name in img_names:
        img_name = img_name.split(".")[0]
        img_path = os.path.join(img_dir, img_name+".tif")
        label_path = os.path.join(label_dir, img_name+"_label.tif")

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        label = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)

        img_width, img_height, _ = img.shape
        # determine the padding length
        if img_width % target_width == 0:
            lw = 0
        else:
            lw = (int(img_width/target_width) + 1) * target_width
            lw = int((lw - img_width)/2)
        if img_height % target_height == 0:
            lh = 0
        else:
            lh = (int(img_height/target_height) + 1) * target_height
            lh = int((lh - img_height) / 2)
        img_padded = cv2.copyMakeBorder(img, lw, lw, lh, lh, cv2.BORDER_CONSTANT, value=label_dict["unknown"][1])
        label_padded = cv2.copyMakeBorder(label, lw, lw, lh, lh, cv2.BORDER_CONSTANT, value=label_dict["unknown"][1])

        num_w = int(img_padded.shape[0]/target_width)
        num_h = int(img_padded.shape[1]/target_height)

        for iw in range(0, num_w):
            for ih in range(0, num_h):
                img_cropped = img_padded[iw*target_width:(iw+1)*target_width, ih*target_height:(ih+1)*target_height, :]
                label_cropped = label_padded[iw*target_width:(iw+1)*target_width, ih*target_height:(ih+1)*target_height, :]

                img_save_path = os.path.join(dataset_dir, "img_RGB", str(num_save)+".tiff")
                label_save_path = os.path.join(dataset_dir, "label_5classes", str(num_save) + ".tiff")

                cv2.imwrite(img_save_path, cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR))
                cv2.imwrite(label_save_path, cv2.cvtColor(label_cropped, cv2.COLOR_RGB2BGR))
                num_save += 1


if __name__ == "__main__":
    # generate dataset for classification (backbone pretraining)  from raw data
    # backbone_dataGen(img_dir, label_dir, dataset_dir="dataset/train", num_sample=2000)
    # generate dataset for segmentation from raw data
    seg_dataGen(img_dir, label_dir, dataset_dir="SegDataset/train", target_size=[224, 224])