import os
import gc
import matplotlib.pyplot as plt
import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms

from final_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------PARAMS------------------------
parser = argparse.ArgumentParser(description='Neopolyp Segmentation Inference')
parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint')
parser.add_argument('--test_dir', type=str, help='Directory path to test images')
parser.add_argument('--mask_dir', type=str, help='Directory path to save predicted masks')
args = parser.parse_args()

# -----------------------DATASET------------------------
TEST_PATH = args.test_dir
TEST_BATCH_SIZE = 8

neopolyp_test_dataset = NeopolypTestDataset(images_path=TEST_PATH)
test_dataloader = DataLoader(neopolyp_test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

# ----------------------CHECKPOINT----------------------
CHECKPOINT_PATH = torch.load(args.checkpoint)
checkpoint = torch.load(CHECKPOINT_PATH)
model_state_dict = checkpoint['model_state_dict']
model = PretrainedUNet(for_inference=True)
model.load_state_dict(model_state_dict)
model = nn.DataParallel(model)

# -------------------_---VISUALIZE----------------------

model.to(device)

images, _, _, _ = next(iter(test_dataloader))
images = images.to(device)

fig, axes = plt.subplots(TEST_BATCH_SIZE, 2, figsize=(16, 12))
axes[0][0].set_title('Image')
axes[0][1].set_title('Predict')

model.eval()
with torch.no_grad():
    for i in range(TEST_BATCH_SIZE):
        axes[i][0].imshow(images[i].permute(1, 2, 0).cpu())
        image = images[i].unsqueeze(0).to(device)
        predict = model(image)
        axes[i][1].imshow(F.one_hot(torch.argmax(predict.squeeze(), 0).cpu()).float())
    
        del predict
        torch.cuda.empty_cache()

plt.tight_layout()
plt.show()

# ----------------------PREDICTION----------------------
PREDICTION_PATH = args.mask_dir
if not os.path.isdir(PREDICTION_PATH):
    os.mkdir(PREDICTION_PATH)

model.eval()
with torch.no_grad():
    for images, paths, heights, weights in test_dataloader:

        for i in range(TEST_BATCH_SIZE):
            image = images[i].unsqueeze(0).to(device)
            predict = model(image)

            image_id = paths[i].split('/')[-1].split('.')[0]
            filename = image_id + ".png"
            mask2img = transforms.Resize((heights[i].item(), weights[i].item()), interpolation=transforms.InterpolationMode.NEAREST)(transforms.ToPILImage()(F.one_hot(torch.argmax(predict.squeeze(), 0)).permute(2, 0, 1).float()))
            mask2img.save(os.path.join(PREDICTION_PATH, filename))
            
            del image
            del predict
            gc.collect()
            torch.cuda.empty_cache()
print("> Prediction saved.")

# --------------------CONVERT TO CSV--------------------
import numpy as np
import pandas as pd
import cv2
import os

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def rle2mask(mask_rle, shape=(3,3)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r

MASK_DIR_PATH = PREDICTION_PATH
dir = MASK_DIR_PATH
res = mask2string(dir)
df = pd.DataFrame(columns=['Id', 'Expected'])
df['Id'] = res['ids']
df['Expected'] = res['strings']

df.to_csv(r'submission.csv', index=False)
print("> Submission saved.")