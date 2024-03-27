import torch
import torch.nn as nn
import torch.nn.functional as F

# class MaskPredictor(nn.Module):
#     def __init__(self):
#         super(MaskPredictor, self).__init__()
        
#         # Encoding layers
#         self.conv1 = nn.Conv2d(1280, 512, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        
#         # Up-sampling layers
#         self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
#         self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 129 = 128 + 1, 1 from mask
#         self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose2d(33, 16, kernel_size=4, stride=2, padding=1)
#         self.deconv5 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
#         self.deconv6 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)
        
#     def forward(self, image_features, foreground_mask):
#         # Encoding
#         x = F.relu(self.conv1(image_features)) # torch.Size([8, 512, 8, 8])
#         x = F.relu(self.conv2(x)) # torch.Size([8, 256, 8, 8])
        
#         # Decoding to get the final mask
#         x = F.relu(self.deconv1(x)) # torch.Size([8, 128, 16, 16])
#         x = F.relu(self.deconv2(x)) # torch.Size([8, 64, 32, 32])
#         x = F.relu(self.deconv3(x)) # torch.Size([8, 32, 64, 64])
        
#         # Up-sampling foreground mask to 64x64
#         mask = F.interpolate(foreground_mask, size=(64, 64), mode='bilinear', align_corners=True)        
#         # Concatenating mask with image features
#         x = torch.cat([x, mask], dim=1)
        
#         x = F.relu(self.deconv4(x)) # torch.Size([8, 16, 128, 128])
#         x = F.relu(self.deconv5(x)) # torch.Size([8, 8, 256, 256])
#         x = torch.sigmoid(self.deconv6(x)) # torch.Size([8, 1, 512, 512])
        
#         return x
    
# # Loss function
# def mask_loss(predicted_mask, gt_mask):
#     criterion = nn.BCELoss()  # Binary Cross Entropy
#     return criterion(predicted_mask, gt_mask)

# # Test
# batch_size = 8
# image_features = torch.randn(batch_size, 1280, 8, 8)
# foreground_mask = torch.randn(batch_size, 1, 512, 512)
# gt_mask = torch.randint(0, 2, (batch_size, 1, 512, 512), dtype=torch.float32)

# model = MaskPredictor()
# output_mask = model(image_features, foreground_mask)

# loss = mask_loss(output_mask, gt_mask)
# print(loss.item())


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MaskPredictor(nn.Module):
#     def __init__(self):
#         super(MaskPredictor, self).__init__()
        
#         # Features fusion
#         self.conv_fuse_1 = nn.Conv2d(320*3, 320, kernel_size=1)
#         self.conv_fuse_2 = nn.Conv2d(640, 320, kernel_size=1)  # changed input channels from 640*3 to 640
#         self.conv_fuse_3 = nn.Conv2d(640, 320, kernel_size=1)  # same here
#         self.conv_fuse_4 = nn.Conv2d(1280, 320, kernel_size=1) # and here
        
#         # Decoder
#         self.deconv1 = nn.ConvTranspose2d(2241, 320, kernel_size=4, stride=2, padding=1) # +1 for the mask channel
#         self.deconv2 = nn.ConvTranspose2d(320, 160, kernel_size=3, stride=1, padding=1)
#         self.deconv3 = nn.ConvTranspose2d(160, 80, kernel_size=4, stride=2, padding=1)
#         self.deconv4 = nn.ConvTranspose2d(80, 40, kernel_size=3, stride=1, padding=1)
#         self.deconv5 = nn.ConvTranspose2d(40, 1, kernel_size=4, stride=2, padding=1)
        
#     def forward(self, features, foreground_mask):
#         # Reshaping and fusing the features to 64x64
#         features_64 = [F.interpolate(f, size=(64, 64), mode='bilinear', align_corners=True) for f in features[:3]]
#         features_64.append(self.conv_fuse_1(torch.cat(features_64, dim=1)))
        
#         # Process 32x32 features
#         features_32 = [self.conv_fuse_2(f) for f in features[3:6]]
#         fused_32 = torch.sum(torch.stack([F.interpolate(f, size=(64, 64), mode='bilinear', align_corners=True) for f in features_32]), dim=0)
#         features_64.append(fused_32)
        
#         # Similar processing for 16x16 and 8x8 features
#         features_16 = [self.conv_fuse_3(f) for f in features[6:9]]
#         fused_16 = torch.sum(torch.stack([F.interpolate(f, size=(64, 64), mode='bilinear', align_corners=True) for f in features_16]), dim=0)
#         features_64.append(fused_16)

#         features_8 = [self.conv_fuse_4(f) for f in features[9:]]
#         fused_8 = torch.sum(torch.stack([F.interpolate(f, size=(64, 64), mode='bilinear', align_corners=True) for f in features_8]), dim=0)
#         features_64.append(fused_8)
        
#         # Up-sampling foreground mask to 64x64
#         mask = F.interpolate(foreground_mask, size=(64, 64), mode='bilinear', align_corners=True)
        
#         # Concatenating mask with image features
#         x = torch.cat(features_64 + [mask], dim=1)
        
#         # Decoding to get the final mask
#         x = F.relu(self.deconv1(x))
#         x = F.relu(self.deconv2(x))
#         x = F.relu(self.deconv3(x))
#         x = F.relu(self.deconv4(x))
#         x = torch.sigmoid(self.deconv5(x))
        
#         return x

# # Example usage:
# model = MaskPredictor()
# batch_size = 2
# features = [torch.randn(batch_size, 320, 64, 64) for _ in range(3)] + \
#            [torch.randn(batch_size, 640, 32, 32) for _ in range(3)] + \
#            [torch.randn(batch_size, 640, 16, 16) for _ in range(3)] + \
#            [torch.randn(batch_size, 1280, 8, 8) for _ in range(3)]

# foreground_mask = torch.randn(batch_size, 1, 512, 512)
# predicted_mask = model(features, foreground_mask)

# # Loss example with a ground truth mask
# gt_mask = torch.randn(batch_size, 512, 512)
# criterion = nn.BCELoss()
# loss = criterion(predicted_mask.squeeze(1), gt_mask)
# print(loss)


import torch.nn as nn
import torch.nn.functional as F

class MaskPredictor(nn.Module):
    def __init__(self):
        super(MaskPredictor, self).__init__()

        # Feature matching
        self.conv_8 = nn.Conv2d(1280, 320, kernel_size=1)
        self.conv_16 = nn.Conv2d(1280, 320, kernel_size=1)
        self.conv_32_1280 = nn.Conv2d(1280, 320, kernel_size=1)
        self.conv_32_640 = nn.Conv2d(640, 320, kernel_size=1)

        # After feature fusion and foreground mask concatenation
        self.conv_combined = nn.Conv2d(4161, 320, kernel_size=3, padding=1)  # 320*7 from features + 1 from mask

        # Upsampling layers to get to 512x512 resolution
        self.deconv1 = nn.ConvTranspose2d(320, 320, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(320, 320, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(320, 320, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(320, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, features, foreground_mask):
        # Match channels and upsample all features to 64x64
        features_64 = [F.interpolate(self.conv_8(f), size=(64, 64), mode='bilinear', align_corners=True) for f in features[:2]]
        features_64 += [F.interpolate(self.conv_16(f), size=(64, 64), mode='bilinear', align_corners=True) for f in features[2:5]]
        features_64.append(F.interpolate(self.conv_32_1280(features[5]), size=(64, 64), mode='bilinear', align_corners=True))
        features_64 += [F.interpolate(self.conv_32_640(f), size=(64, 64), mode='bilinear', align_corners=True) for f in features[6:8]]
        features_64 += features[8:]

        # Concatenate all features together
        x = torch.cat(features_64, dim=1)

        # Upsample foreground mask to 64x64
        mask = F.interpolate(foreground_mask, size=(64, 64), mode='bilinear', align_corners=True)

        # Concatenate the mask
        x = torch.cat([x, mask], dim=1)

        # Process through the rest of the network
        x = F.relu(self.conv_combined(x))
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.deconv4(x))

        return x


model = MaskPredictor()
batch_size = 2
features = [torch.randn(batch_size, 1280, 8, 8) for _ in range(2)] + \
           [torch.randn(batch_size, 1280, 16, 16) for _ in range(3)] + \
           [torch.randn(batch_size, 1280, 32, 32) for _ in range(1)] + \
           [torch.randn(batch_size, 640, 32, 32) for _ in range(2)]+ \
           [torch.randn(batch_size, 640, 64, 64) for _ in range(1)]+ \
           [torch.randn(batch_size, 320, 64, 64) for _ in range(3)]



foreground_mask = torch.randn(batch_size, 1, 512, 512)
predicted_mask = model(features, foreground_mask)

# Loss example with a ground truth mask
gt_mask = torch.randn(batch_size, 512, 512)
criterion = nn.BCELoss()
loss = criterion(predicted_mask.squeeze(1), gt_mask)
print(loss)
