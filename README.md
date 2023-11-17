# unet-for-neopolyp-segmentation

## Work by: Doan The Vinh - 20210940
Score = 0.71927

## Instruction

**0. Access and obtain the checkpoint**
```
import requests
import os

drive_url = 'https://drive.google.com/file/d/13bUu1Nwes9xUXcUTlZFtOpFY1pt3hXK1/view?usp=sharing'
save_dir = '/kaggle/working/'

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
!pip install -r requirements.txt
```

**3. Run inference**
```
!python /kaggle/working/BKAI_Polyp/infer.py --checkpoint '/kaggle/working/model_checkpoint.pth' --test_dir '/kaggle/input/bkai-igh-neopolyp/test/test' --mask_dir '/kaggle/working/predicted_mask'
```