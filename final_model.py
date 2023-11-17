import os

from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

import torchvision.transforms.functional as TF
from torchvision import transforms

import segmentation_models_pytorch as smp

# --------------------TEST DATASET--------------------

class NeopolypTestDataset(Dataset):
    def __init__(self, images_path):
        super(NeopolypTestDataset, self).__init__()
        
        images_list = os.listdir(images_path)
        images_list = [images_path + image_name for image_name in images_list]
        
        self.images_list = images_list
    
    def transform(self, image):
        # Resize
        # resize = transforms.Resize(size=(640, 896), interpolation=transforms.InterpolationMode.BILINEAR)
        resize = transforms.Resize(size=(768, 1024), interpolation=transforms.InterpolationMode.BILINEAR)
        image = resize(image)      
        
        # Transform to tensor
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        
        # Normalize
        normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        image = normalize(image)
        
        return image
    
    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        data = Image.open(img_path)
        h = data.size[1]
        w = data.size[0]
        data = self.transform(data)
        
        return data, img_path, h, w
    
    def __len__(self):
        return len(self.images_list)

# -----------------------MODEL------------------------

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        
        mid_channels = int(out_channels/2)
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.6)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        
        next_layer = self.max_pool(x)
        skip_layer = x
        
        return next_layer, skip_layer
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        mid_channels = int(3 * out_channels/2)
        
        self.transposed_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv2d(2 * out_channels, mid_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.elu = nn.ELU() 
        self.dropout = nn.Dropout(p=0.6)
    
    def forward(self, x, skip_layer):
        x = self.transposed_conv(x)
        
        # Concatenate channel-wise (1st dim: batch size, 2nd dim: channel,...)
        x = torch.cat([x, skip_layer], axis=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        
        return x

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same')
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.6)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x

class AttentionBlock(nn.Module):
    def __init__(self, gate_channels, feature_channels, intermediate_channels):
        super(AttentionBlock, self).__init__()
        
        self.gate_transform = nn.Sequential(
            nn.Conv2d(gate_channels, intermediate_channels, kernel_size=1,stride=2,padding=0,bias=True),
            nn.BatchNorm2d(intermediate_channels)
            )
        
        self.feature_transform = nn.Sequential(
            nn.Conv2d(feature_channels, intermediate_channels, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(intermediate_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(intermediate_channels, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self, g, x):
        g_out = self.gate_transform(g)
        x_out = self.feature_transform(x)
        psi = self.relu(g_out + x_out)
        psi = self.psi(psi)
        
        psi = F.interpolate(psi, scale_factor=2.0, mode='bilinear')
        return g*psi

class UNet(nn.Module):
    
    def __init__(self, for_inference=False,
                 train_dataloader=None, valid_dataloader=None,
                 loss_function=None, optimizer=None,
                 learning_rate_scheduler=None, learning_rate=None,
                 num_epochs=None, num_classes=3):
        
        super(UNet, self).__init__()
        
        if not for_inference:
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader
            self.loss_function = loss_function
            self.num_epochs = num_epochs

        # Encoder blocks
        self.enc1 = EncoderBlock(3, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)
        self.enc5 = EncoderBlock(256, 512)

        # Bottleneck block
        self.bottleneck = BottleneckBlock(512, 1024)

        # Decoder blocks
        self.dec1 = DecoderBlock(1024, 512)
        self.dec2 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec4 = DecoderBlock(128, 64)
        self.dec5 = DecoderBlock(64, 32)
        
        # Attention blocks
        self.att1 = AttentionBlock(512, 1024, 768)
        self.att2 = AttentionBlock(256, 512, 384)
        self.att3 = AttentionBlock(128, 256, 192)
        self.att4 = AttentionBlock(64, 128, 96)
        self.att5 = AttentionBlock(32, 64, 48)

        # 1x1 convolution
        self.out = nn.Conv2d(32, num_classes, kernel_size=1, padding='same')
        
        if not for_inference:
            self.optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=1e-05)
            self.learning_rate_scheduler = learning_rate_scheduler(self.optimizer, step_size=12, gamma=0.6)
            self.initialize_weights()
    
    def forward(self, image):
        n1, s1 = self.enc1(image)
        n2, s2 = self.enc2(n1)
        n3, s3 = self.enc3(n2)
        n4, s4 = self.enc4(n3)
        n5, s5 = self.enc5(n4)
        
        btn = self.bottleneck(n5)
        
        a_s5 = self.att1(s5, btn)
        n6 = self.dec1(btn, a_s5)
        
        a_s4 = self.att2(s4, n6)
        n7 = self.dec2(n6, a_s4)
        
        a_s3 = self.att3(s3, n7)
        n8 = self.dec3(n7, a_s3)
        
        a_s2 = self.att4(s2, n8)
        n9 = self.dec4(n8, a_s2)
        
        a_s1 = self.att5(s1, n9)
        n10 = self.dec5(n9, a_s1)
        
        output = self.out(n10)
        
        return output

class PretrainedUNet(smp.Unet):
    
    def __init__(self, for_inference=False,
                 train_dataloader=None, valid_dataloader=None,
                 loss_function=None, optimizer=None,
                 learning_rate_scheduler=None, learning_rate=None,
                 num_epochs=None, num_classes=3,
                 # For pre-trained model
                 encoder_name="resnet34", encoder_depth=5, encoder_weights="imagenet",
                 decoder_channels=[512, 256, 128, 64, 32], decoder_attention_type='scse'):
        
        # Using the pretrained model
        super(PretrainedUNet, self).__init__(
            encoder_name=encoder_name,
            encoder_depth=encoder_depth,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_attention_type=decoder_attention_type,
            classes=num_classes
        )
        
        if not for_inference:
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader
            self.loss_function = loss_function
            self.num_epochs = num_epochs
            self.optimizer = optimizer(self.parameters(), lr=learning_rate, weight_decay=1e-05)
            self.learning_rate_scheduler = learning_rate_scheduler(self.optimizer, step_size=12, gamma=0.6)
