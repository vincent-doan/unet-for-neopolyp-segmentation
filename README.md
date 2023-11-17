# unet-for-neopolyp-segmentation

## Work by: Doan The Vinh - 20210940
Score = 0.71927

## Instruction

**0. Access and obtain the checkpoint**
```
import requests
import os

drive_url = 'https://drive.google.com/file/d/1sX0ZSTyMOAR1lz8eddapJhn5jNrAUXAt/view?usp=sharing'
save_dir = '/kaggle/working/checkpoint/'
os.mkdir(save_dir)

response = requests.get(drive_url)

with open(os.path.join(save_dir, 'model_checkpoint.pth'), 'wb') as f:
    f.write(response.content)
```

**1. Clone the repository**
```
!git clone https://github.com/vincent-doan/unet-for-neopolyp-segmentation
```

**2. Install the necessary packages**
```
!pip install -q -r unet-for-neopolyp-segmentation/requirements.txt
```

**3. Run inference**
```
!python /kaggle/working/unet-for-neopolyp-segmentation/infer.py --checkpoint '/kaggle/working/checkpoint/model_checkpoint.pth' --test_dir '/kaggle/input/bkai-igh-neopolyp/test/test' --mask_dir '/kaggle/working/predicted_mask'
```