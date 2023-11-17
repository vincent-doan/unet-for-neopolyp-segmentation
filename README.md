# unet-for-neopolyp-segmentation

## Work by: Doan The Vinh - 20210940
Score = 0.7191

## Instruction

**0. Access and obtain the checkpoint**
```
!pip -q install gdown
```

```
import gdown
import os

url = 'https://drive.google.com/uc?id=1sX0ZSTyMOAR1lz8eddapJhn5jNrAUXAt'

save_dir = '/kaggle/working/checkpoint/'
os.mkdir(save_dir)

output = os.path.join(save_dir, 'model_checkpoint.pth')
gdown.download(url, output, quiet=False)
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
!python /kaggle/working/unet-for-neopolyp-segmentation/infer.py --checkpoint '/kaggle/working/checkpoint/model_checkpoint.pth' --test_dir '/kaggle/input/bkai-igh-neopolyp/test/test/' --mask_dir '/kaggle/working/predicted_mask/'
```