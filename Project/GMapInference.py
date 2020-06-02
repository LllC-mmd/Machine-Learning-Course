import timeit
import numpy as np
import torch
from torchvision import transforms

from SegModel import *
from SegLoss import *
from dataSet import *
from train_config import *

import gdal
import cv2
from PIL import Image


label_dict = {"built-up": (0, [255, 0, 0]),
              "farmland": (1, [0, 255, 0]),
              "forest": (2, [0, 255, 255]),
              "meadow": (3, [255, 255, 0]),
              "water": (4, [0, 0, 255]),
              "unknown": (5, [0, 0, 0])}

GID_FROMGLC = {0: ("Impervious surface", 80),
               1: ("Cropland", 10),
               2: ("Forest", 20),
               3: ("Grassland", 30),
               4: ("Water", 60)}

func_idMap = np.vectorize(lambda x: GID_FROMGLC[x][1])

map_file = "GoogleMapData/Center/test.tif"

target_size = 224

img = cv2.cvtColor(cv2.imread(map_file), cv2.COLOR_BGR2RGB)
img_height, img_width, _ = img.shape

if img_height % target_size == 0:
    lh1 = lh2 = 0
else:
    lh = (int(img_height / target_size) + 1) * target_size
    if (lh - img_height)%2 == 0:
        lh1 = lh2 = int((lh - img_height) / 2)
    else:
        lh1 = int((lh - img_height) / 2)
        lh2 = lh1 + 1

if img_width % target_size == 0:
    lw1 = lw2 = 0
else:
    lw = (int(img_width / target_size) + 1) * target_size
    if (lw - img_width)%2 == 0:
        lw1 = lw2 = int((lw - img_width) / 2)
    else:
        lw1 = int((lw - img_width) / 2)
        lw2 = lw1 + 1

img_padded = cv2.copyMakeBorder(img, lh1, lh2, lw1, lw2, cv2.BORDER_CONSTANT, value=label_dict["unknown"][1])

num_h = int(img_padded.shape[0] / target_size)
num_w = int(img_padded.shape[1] / target_size)

trained_record = "Result-save/Unet_best.pth.tar"
#model = LUSegDeepLab("AttResNet", num_plane=2048, output_stride=8, num_classes=5, pretrained_backbone="Result-save/pretrained_SEResAttentionNet.pt")
model = LUSegUNet(num_channel=512, num_classes=5)
#model = LUSegUNet_IN(num_channel=512, num_classes=5)

'''
total_num = sum(p.numel() for p in model.parameters())
trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total parameter of mode: ", total_num, " Trainable parameter of model: ", trainable_num)
'''

checkpoint = torch.load(trained_record, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint["state_dict"])
model.eval()

in_trans = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

LU_array = np.zeros((img_padded.shape[0], img_padded.shape[1]))
for ih in range(0, num_h):
    for iw in range(0, num_w):
        start = timeit.default_timer()
        img_cropped = img_padded[ih*target_size:(ih+1)*target_size, iw*target_size:(iw+1)*target_size, : ]
        img_tensor = Image.fromarray(img_cropped, mode="RGB")
        img_tensor = in_trans(img_tensor)
        if torch.cuda.is_available():
            image, target = img_tensor.cuda(), img_tensor.cuda()
        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0))
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1).squeeze(0)
        LU_array[ih*target_size:(ih+1)*target_size, iw*target_size:(iw+1)*target_size] = func_idMap(pred)
        stop = timeit.default_timer()
        print('Run time of {0}-{1} : {2:.4f}'.format(ih, iw, stop - start))
        # np.savetxt("GoogleMapData/{0}-{1}.txt".format(ih, iw), pred, fmt="%d")

LU_array = LU_array[lh1:-lh2, lw1:-lw2]

LU_file = "GoogleMapData/Center/test_Unet_label.tif"
driver = gdal.GetDriverByName("GTiff")
LU_ds = driver.Create(LU_file, LU_array.shape[1], LU_array.shape[0], 1, gdal.GDT_Int16)
LU_ds.GetRasterBand(1).WriteArray(LU_array)

ds = gdal.Open(map_file)
geo_trans = ds.GetGeoTransform()
proj = ds.GetProjection()
LU_ds.SetGeoTransform(geo_trans)
LU_ds.SetProjection(proj)
LU_ds.FlushCache()
LU_ds = None
