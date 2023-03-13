# #-------------------------------------------1st Version - Large Latent Dimension - Two-branches Model:ResNet50 + PointNet----------------------------------------
# import torch
# import torch.nn.functional as F
# from torch import nn
# from PIL import Image
# from torchvision import datasets, transforms
# from torchvision import models


# class PointNetEncoder(nn.Module):
#     def __init__(self, zdim, input_dim=6):
#         super().__init__()
#         # -----------------PointNet encoder-----------------
#         self.zdim = zdim
#         self.conv1 = nn.Conv1d(input_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 128, 1)
#         self.conv3 = nn.Conv1d(128, 512, 1)
#         self.conv4 = nn.Conv1d(512, 2048, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.bn4 = nn.BatchNorm1d(2048)

#         # Mapping to [c], cmean
#         self.fc1_m = nn.Linear(2048, 1024)
#         self.fc2_m = nn.Linear(1024, 512)
#         self.fc3_m = nn.Linear(512, zdim)
#         self.fc_bn1_m = nn.BatchNorm1d(1024)
#         self.fc_bn2_m = nn.BatchNorm1d(512)
#         self.fc_bn3_m = nn.BatchNorm1d(zdim)

#         # Mapping to [c], cmean
#         self.fc1_v = nn.Linear(2048, 1024)
#         self.fc2_v = nn.Linear(1024, 512)
#         self.fc3_v = nn.Linear(512, zdim)
#         self.fc_bn1_v = nn.BatchNorm1d(1024)
#         self.fc_bn2_v = nn.BatchNorm1d(512)
#         self.fc_bn3_v = nn.BatchNorm1d(zdim)


#         # -----------------ResNet50 encoder-----------------
#         self.model = models.resnet50(pretrained=True)
#         #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
#         self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.model.fc = nn.Linear(2048, zdim)
#         self.model.fc_bn = nn.BatchNorm1d(zdim)


#         # -----------------Fusion MLP-----------------
#         # Fusion two latent codes together
#         self.fc1_con = torch.nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024

        

    


#     def forward(self, x, img):
#         # -----------------Shape Latent Code from PointNet-----------------
#                                                    # ([8, 52000, 6])
#         x = x.transpose(1, 2)                      # ([8, 6, 52000])
#         x = F.relu(self.bn1(self.conv1(x)))        # ([8, 128, 52000])
#         x = F.relu(self.bn2(self.conv2(x)))        # ([8, 128, 52000])
#         x = F.relu(self.bn3(self.conv3(x)))        # ([8, 256, 52000])
#         x = self.bn4(self.conv4(x))                # ([8, 512, 52000])
#         x = torch.max(x, 2, keepdim=True)[0]       # ([8, 512, 1])
#         x = x.view(-1, 2048)                        # ([8, 512])

#         m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # ([8, 256])
#         m1 = F.relu(self.fc_bn2_m(self.fc2_m(m1))) # ([8, 128])
#         m1 = F.relu(self.fc_bn3_m(self.fc3_m(m1))) # ([8, 64])
#         # m1 = self.fc3_m(m1) # 256 dimension
#         v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
#         v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
#         v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
#         # v = self.fc3_v(v)
#         # v = 0


#         # -----------------Image Latent Code from ResNet50-----------------
#         img = self.model.conv1(img)    # ([8, 64, 240, 320])
#         img = self.model.bn1(img)      # ([8, 64, 240, 320])
#         img = self.model.relu(img)     # ([8, 64, 240, 320])
#         img = self.model.maxpool(img)  # ([8, 64, 120, 160])
#         img = self.model.layer1(img)   # ([8, 256, 120, 160])
#         img = self.model.layer2(img)   # ([8, 512, 60, 80])
#         img = self.model.layer3(img)   # ([8, 1024, 30, 40])
#         img = self.model.layer4(img)   # ([8, 2048, 15, 20])
#         img = self.model.avgpool(img)  # ([8, 2048, 1, 1])
#         img = img.view(img.size(0), -1)# ([8, 2048])
#         # m2 = self.model.fc(img)
#         # m2 = self.model.fc_bn(m2)
#         # m2 = F.relu(m2)
#         m2 = F.relu(self.model.fc_bn(self.model.fc(img))) # ([8, 64])


#         # ----------------------------------Fusion MLP----------------------------------
#         # MLP to combine the two code and make the dimensions equal to zdim
#         m = torch.cat([m1, m2], dim=-1) # (B, zdim+zdim)
#         m = self.fc1_con(m) # convert ResNet50 Latent code from 2*zdim to zdim
#         v = torch.cat([v, m2], dim=-1) # (B, zdim+zdim)
#         v = self.fc1_con(v) # convert ResNet50 Latent code from 2*zdim to zdim


#         # Returns both mean and logvariance, just ignore the latter in deteministic cases.
#         return m, v
    






# #-------------------------------------------Generator - Two-branches Model:ResNet50 + PointNet----------------------------------------
# import torch
# import torch.nn.functional as F
# from torch import nn
# from PIL import Image
# from torchvision import datasets, transforms
# from torchvision import models


# class PointNetEncoder(nn.Module):
#     def __init__(self, zdim, input_dim=6):
#         super().__init__()
#         # -----------------PointNet encoder-----------------
#         self.zdim = zdim
#         self.conv1 = nn.Conv1d(input_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 128, 1)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.conv4 = nn.Conv1d(256, 512, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(512)

#         # Mapping to [c], cmean
#         self.fc1_m = nn.Linear(512, 256)
#         self.fc2_m = nn.Linear(256, 128)
#         self.fc3_m = nn.Linear(128, zdim)
#         self.fc_bn1_m = nn.BatchNorm1d(256)
#         self.fc_bn2_m = nn.BatchNorm1d(128)
#         self.fc_bn3_m = nn.BatchNorm1d(zdim)

#         # Mapping to [c], cmean
#         self.fc1_v = nn.Linear(512, 256)
#         self.fc2_v = nn.Linear(256, 128)
#         self.fc3_v = nn.Linear(128, zdim)
#         self.fc_bn1_v = nn.BatchNorm1d(256)
#         self.fc_bn2_v = nn.BatchNorm1d(128)
#         self.fc_bn3_v = nn.BatchNorm1d(zdim)


#         # -----------------ResNet50 encoder-----------------
#         self.model = models.resnet50(pretrained=True)
#         #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
#         self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.model.fc = nn.Linear(2048, zdim)
#         self.model.fc_bn = nn.BatchNorm1d(zdim)


#         # -----------------Fusion MLP-----------------
#         # Fusion two latent codes together
#         self.fc1_con = torch.nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024

        

    


#     def forward(self, x, img):
#         # -----------------Shape Latent Code from PointNet-----------------
#                                                    # ([8, 52000, 6])
#         x = x.transpose(1, 2)                      # ([8, 6, 52000])
#         x = F.relu(self.bn1(self.conv1(x)))        # ([8, 128, 52000])
#         x = F.relu(self.bn2(self.conv2(x)))        # ([8, 128, 52000])
#         x = F.relu(self.bn3(self.conv3(x)))        # ([8, 256, 52000])
#         x = self.bn4(self.conv4(x))                # ([8, 512, 52000])
#         x = torch.max(x, 2, keepdim=True)[0]       # ([8, 512, 1])
#         x = x.view(-1, 512)                        # ([8, 512])

#         m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # ([8, 256])
#         m1 = F.relu(self.fc_bn2_m(self.fc2_m(m1))) # ([8, 128])
#         m1 = F.relu(self.fc_bn3_m(self.fc3_m(m1))) # ([8, 64])
#         # m1 = self.fc3_m(m1) # 256 dimension
#         v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
#         v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
#         v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
#         # v = self.fc3_v(v)
#         # v = 0


#         # -----------------Image Latent Code from ResNet50-----------------
#         img = self.model.conv1(img)    # ([8, 64, 240, 320])
#         img = self.model.bn1(img)      # ([8, 64, 240, 320])
#         img = self.model.relu(img)     # ([8, 64, 240, 320])
#         img = self.model.maxpool(img)  # ([8, 64, 120, 160])
#         img = self.model.layer1(img)   # ([8, 256, 120, 160])
#         img = self.model.layer2(img)   # ([8, 512, 60, 80])
#         img = self.model.layer3(img)   # ([8, 1024, 30, 40])
#         img = self.model.layer4(img)   # ([8, 2048, 15, 20])
#         img = self.model.avgpool(img)  # ([8, 2048, 1, 1])
#         img = img.view(img.size(0), -1)# ([8, 2048])
#         # m2 = self.model.fc(img)
#         # m2 = self.model.fc_bn(m2)
#         # m2 = F.relu(m2)
#         m2 = F.relu(self.model.fc_bn(self.model.fc(img))) # ([8, 64])


#         # ----------------------------------Fusion MLP----------------------------------
#         # MLP to combine the two code and make the dimensions equal to zdim
#         m = torch.cat([m1, m2], dim=-1) # (B, zdim+zdim)
#         m = self.fc1_con(m) # convert ResNet50 Latent code from 2*zdim to zdim
#         v = torch.cat([v, m2], dim=-1) # (B, zdim+zdim)
#         v = self.fc1_con(v) # convert ResNet50 Latent code from 2*zdim to zdim


#         # Returns both mean and logvariance, just ignore the latter in deteministic cases.
#         return m, v
    





# #-------------------------------------------Two-branches Model:ResNet50 + PointNet----------------------------------------
# import torch
# import torch.nn.functional as F
# from torch import nn
# from PIL import Image
# from torchvision import datasets, transforms
# from torchvision import models


# class PointNetEncoder(nn.Module):
#     def __init__(self, zdim, input_dim=6):
#         super().__init__()
#         # -----------------PointNet encoder-----------------
#         self.zdim = zdim
#         self.conv1 = nn.Conv1d(input_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 128, 1)
#         self.conv3 = nn.Conv1d(128, 256, 1)
#         self.conv4 = nn.Conv1d(256, 512, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.bn3 = nn.BatchNorm1d(256)
#         self.bn4 = nn.BatchNorm1d(512)

#         # Mapping to [c], cmean
#         self.fc1_m = nn.Linear(512, 256)
#         self.fc2_m = nn.Linear(256, 128)
#         self.fc3_m = nn.Linear(128, zdim)
#         self.fc_bn1_m = nn.BatchNorm1d(256)
#         self.fc_bn2_m = nn.BatchNorm1d(128)
#         self.fc_bn3_m = nn.BatchNorm1d(zdim)

#         # Mapping to [c], cmean
#         self.fc1_v = nn.Linear(512, 256)
#         self.fc2_v = nn.Linear(256, 128)
#         self.fc3_v = nn.Linear(128, zdim)
#         self.fc_bn1_v = nn.BatchNorm1d(256)
#         self.fc_bn2_v = nn.BatchNorm1d(128)
#         self.fc_bn3_v = nn.BatchNorm1d(zdim)


#         # -----------------ResNet50 encoder-----------------
#         self.model = models.resnet50(pretrained=True)
#         #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
#         self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.model.fc = nn.Linear(2048, zdim)
#         self.model.fc_bn = nn.BatchNorm1d(zdim)


#         # -----------------Fusion MLP-----------------
#         # Fusion two latent codes together
#         self.fc1_con = torch.nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024

        

    


#     def forward(self, x, img):
#         # -----------------Shape Latent Code from PointNet-----------------
#                                                    # ([8, 52000, 6])
#         x = x.transpose(1, 2)                      # ([8, 6, 52000])
#         x = F.relu(self.bn1(self.conv1(x)))        # ([8, 128, 52000])
#         x = F.relu(self.bn2(self.conv2(x)))        # ([8, 128, 52000])
#         x = F.relu(self.bn3(self.conv3(x)))        # ([8, 256, 52000])
#         x = self.bn4(self.conv4(x))                # ([8, 512, 52000])
#         x = torch.max(x, 2, keepdim=True)[0]       # ([8, 512, 1])
#         x = x.view(-1, 512)                        # ([8, 512])

#         m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # ([8, 256])
#         m1 = F.relu(self.fc_bn2_m(self.fc2_m(m1))) # ([8, 128])
#         m1 = F.relu(self.fc_bn3_m(self.fc3_m(m1))) # ([8, 64])
#         # m1 = self.fc3_m(m1) # 256 dimension
#         v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
#         v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
#         v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
#         v = self.fc3_v(v)
#         # v = 0


#         # -----------------Image Latent Code from ResNet50-----------------
#         img = self.model.conv1(img)    # ([8, 64, 240, 320])
#         img = self.model.bn1(img)      # ([8, 64, 240, 320])
#         img = self.model.relu(img)     # ([8, 64, 240, 320])
#         img = self.model.maxpool(img)  # ([8, 64, 120, 160])
#         img = self.model.layer1(img)   # ([8, 256, 120, 160])
#         img = self.model.layer2(img)   # ([8, 512, 60, 80])
#         img = self.model.layer3(img)   # ([8, 1024, 30, 40])
#         img = self.model.layer4(img)   # ([8, 2048, 15, 20])
#         img = self.model.avgpool(img)  # ([8, 2048, 1, 1])
#         img = img.view(img.size(0), -1)# ([8, 2048])
#         # m2 = self.model.fc(img)
#         # m2 = self.model.fc_bn(m2)
#         # m2 = F.relu(m2)
#         m2 = F.relu(self.model.fc_bn(self.model.fc(img))) # ([8, 64])


#         # ----------------------------------Fusion MLP----------------------------------
#         # MLP to combine the two code and make the dimensions equal to zdim
#         m = torch.cat([m1, m2], dim=-1) # (B, zdim+zdim)
#         m = self.fc1_con(m) # convert ResNet50 Latent code from 2*zdim to zdim


#         # Returns both mean and logvariance, just ignore the latter in deteministic cases.
#         return m, v
    







# # -------------------------------------------Old Skip-Connection - Two-branches Model:ResNet50 + PointNet----------------------------------------
# import torch
# import torch.nn.functional as F
# from torch import nn
# from PIL import Image
# from torchvision import datasets, transforms
# from torchvision import models


# class PointNetEncoder(nn.Module):
#     def __init__(self, zdim, input_dim=6):
#         super().__init__()
#         # -----------------PointNet encoder-----------------
#         self.zdim = zdim
#         self.conv1 = nn.Conv1d(input_dim, 128, 1)
#         self.conv2 = nn.Conv1d(128, 256, 1)
#         self.conv3 = nn.Conv1d(256, 512, 1)
#         self.conv4 = nn.Conv1d(256, 2048, 1)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.bn3 = nn.BatchNorm1d(512)
#         self.bn4 = nn.BatchNorm1d(2048)

#         # # Mapping to [c], cmean
#         # self.fc1_m = nn.Linear(512, zdim)
#         # self.fc_bn1_m = nn.BatchNorm1d(zdim)
#         # Mapping to [c], cmean
#         self.fc1_m = nn.Linear(2048, 1024)
#         self.fc2_m = nn.Linear(1024, 512)
#         self.fc3_m = nn.Linear(512, zdim)
#         self.fc_bn1_m = nn.BatchNorm1d(1024)
#         self.fc_bn2_m = nn.BatchNorm1d(512)
#         self.fc_bn3_m = nn.BatchNorm1d(zdim)

#         self.fc1_v = nn.Linear(2048, 1024)
#         self.fc2_v = nn.Linear(1024, 512)
#         self.fc3_v = nn.Linear(512, zdim)
#         self.fc_bn1_v = nn.BatchNorm1d(1024)
#         self.fc_bn2_v = nn.BatchNorm1d(512)
#         self.fc_bn3_v = nn.BatchNorm1d(zdim)


#         # -----------------ResNet50 encoder-----------------
#         self.model = models.resnet50(pretrained=True)
#         #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
#         self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.model.fc = nn.Linear(2048, zdim)
#         self.model.fc_bn = nn.BatchNorm1d(zdim)


#         # -----------------Skip-Connection - Latent Code Fusion MLP-----------------
#         # Fusion two latent codes together
#         self.fc1_latent = nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024


#         # # !!!!!!!!!!!!!!Need implementation!!!!!!!!!!!!!!!!!!
#         # # -----------------Skip-Connection - Feature Map Fusion MLP-----------------
#         # # Fusion two latent codes together
#         # self.fc1_skipCon = nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024

        

    


#     def forward(self, x, img):
#         # -----------------Shape Latent Code from PointNet-----------------
#                                                    # ([8, 52000, 6])
#         x = x.transpose(1, 2)                      # ([8, 6, 52000])
#         x = F.relu(self.bn1(self.conv1(x)))        # ([8, 128, 52000])
#         x_skip1 = F.relu(self.bn2(self.conv2(x)))        # ([8, 256, 52000])
#         x_skip2 = F.relu(self.bn3(self.conv3(x_skip1)))        # ([8, 512, 52000])
#         x_skip3 = self.bn4(self.conv4(x_skip2))                # ([8, 2048, 52000])
#         x = torch.max(x_skip3, 2, keepdim=True)[0]       # ([8, 2048, 1])
#         x = x.view(-1, 2048)                        # ([8, 2048])

#         m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # ([8, 1024])
#         m1 = F.relu(self.fc_bn2_m(self.fc2_m(m1))) # ([8, 512])
#         m1 = F.relu(self.fc_bn3_m(self.fc3_m(m1))) # ([8, zdim])
#         # m1 = self.fc3_m(m1) # 256 dimension
#         v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
#         v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
#         v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
#         # v = self.fc3_v(v)


#         # -----------------Image Latent Code from ResNet50-----------------
#         img = self.model.conv1(img)    # ([8, 64, 240, 320])
#         img = self.model.bn1(img)      # ([8, 64, 240, 320])
#         img = self.model.relu(img)     # ([8, 64, 240, 320])
#         img = self.model.maxpool(img)  # ([8, 64, 120, 160])
#         img_skip1 = self.model.layer1(img)   # ([8, 256, 120, 160])
#         img_skip2 = self.model.layer2(img_skip1)   # ([8, 512, 60, 80])
#         img_skip3 = self.model.layer3(img_skip2)   # ([8, 1024, 30, 40])
#         img = self.model.layer4(img_skip3)   # ([8, 2048, 15, 20])
#         img = self.model.avgpool(img)  # ([8, 2048, 1, 1])
#         img = img.view(img.size(0), -1)# ([8, 2048])
#         # m2 = self.model.fc(img)
#         # m2 = self.model.fc_bn(m2)
#         # m2 = F.relu(m2)
#         m2 = F.relu(self.model.fc_bn(self.model.fc(img))) # ([8, zdim])


#         # ----------------------------------Skip-Connection - Latent Code Fusion MLP---------------------------------
#         # MLP to combine the two code and make the dimensions equal to zdim
#         m = torch.cat([m1, m2], dim=-1) # (B, zdim+zdim)
#         m = self.fc1_latent(m) # convert ResNet50 Latent code from 2*zdim to zdim

#         # # !!!!!!!!!!!!!!Need implementation!!!!!!!!!!!!!!!!!!
#         # # ----------------------------------Skip-Connection - Feature Map Fusion MLP----------------------------------
#         # # MLP to combine the two code and make the dimensions equal to zdim
#         # m = torch.cat([m1, m2], dim=-1) # (B, zdim+zdim)
#         # m = self.fc1_con(m) # convert ResNet50 Latent code from 2*zdim to zdim


#         # Returns both mean and logvariance, just ignore the latter in deteministic cases.
#         return m, v











#-------------------------------------------New Skip-Connection - Two-branches Model:ResNet50 + PointNet----------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models
import copy


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_downsample, input_dim=6):
        super().__init__()
        # -----------------PointNet encoder-----------------
        self.zdim = zdim
        self.input_downsample = input_downsample
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(2048)

        # # Mapping to [c], cmean
        # self.fc1_m = nn.Linear(512, zdim)
        # self.fc_bn1_m = nn.BatchNorm1d(zdim)
        # Mapping to [c], cmean
        # self.fc1_m = nn.Linear(2048, 1024)
        # self.fc2_m = nn.Linear(1024, 512)
        # self.fc3_m = nn.Linear(512, zdim)
        # self.fc_bn1_m = nn.BatchNorm1d(1024)
        # self.fc_bn2_m = nn.BatchNorm1d(512)
        # self.fc_bn3_m = nn.BatchNorm1d(zdim)

        # self.fc1_v = nn.Linear(2048, 1024)
        # self.fc2_v = nn.Linear(1024, 512)
        # self.fc3_v = nn.Linear(512, zdim)
        # self.fc_bn1_v = nn.BatchNorm1d(1024)
        # self.fc_bn2_v = nn.BatchNorm1d(512)
        # self.fc_bn3_v = nn.BatchNorm1d(zdim)


        # -----------------ResNet50 encoder-----------------
        self.model = models.resnet50(pretrained=True)
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
        # self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.model.fc = nn.Linear(2048, zdim)
        # self.model.fc_bn = nn.BatchNorm1d(zdim)


        # -----------------Attention Block-----------------
        # Form query, key and value
        self.Wq1 = nn.Linear(256, 256)  
        self.Wk1 = nn.Linear(256, 256)  
        self.Wv1 = nn.Linear(256, 256)  
        self.Wq2 = nn.Linear(512, 512)  
        self.Wk2 = nn.Linear(512, 512)   
        self.Wv2 = nn.Linear(512, 512) 
        self.Wq3 = nn.Linear(2048, 2048)  
        self.Wk3 = nn.Linear(2048, 2048)  
        self.Wv3 = nn.Linear(2048, 2048)  
        


        # # -----------------Latent Code - maxpooling the last fused feature map-----------------
        # Maxpool the last fused feature map
        # self.fc1_skipCon = nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024

        

    


    def forward(self, x, img):
        # -----------------Shape Latent Code from PointNet-----------------
        x_portion = x
        for i in range(self.input_downsample-1):
            x = torch.cat((x,x_portion), dim=1)
                                                   # ([B, 52000, 6])
        x = x.transpose(1, 2)                      # ([B, 6, 52000])
        x = F.relu(self.bn1(self.conv1(x)))        # ([B, 128, 52000])
        x_skip1 = F.relu(self.bn2(self.conv2(x)))        # ([B, 256, 52000])
        x_skip2 = F.relu(self.bn3(self.conv3(x_skip1)))        # ([B, 512, 52000])
        x_skip3 = self.bn4(self.conv4(x_skip2))                # ([B, 2048, 52000])
        # x = torch.max(x_skip3, 2, keepdim=True)[0]       # ([B, 2048, 1])
        # x = x.view(-1, 2048)                        # ([B, 2048])

        # m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # ([B, 1024])
        # m1 = F.relu(self.fc_bn2_m(self.fc2_m(m1))) # ([B, 512])
        # m1 = F.relu(self.fc_bn3_m(self.fc3_m(m1))) # ([B, zdim])
        # # m1 = self.fc3_m(m1) # 256 dimension
        # v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        # v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        # v = F.relu(self.fc_bn3_v(self.fc3_v(v)))
        # # v = self.fc3_v(v)



        # -----------------Image Latent Code from ResNet50-----------------
        img = self.model.conv1(img)    # ([B, 64, 240, 320])
        img = self.model.bn1(img)      # ([B, 64, 240, 320])
        img = self.model.relu(img)     # ([B, 64, 240, 320])
        img = self.model.maxpool(img)  # ([B, 64, 120, 160])
        img_skip1 = self.model.layer1(img)   # ([B, 256, 120, 160])
        img_skip2 = self.model.layer2(img_skip1)   # ([B, 512, 60, 80])
        img = self.model.layer3(img_skip2)   # ([B, 1024, 30, 40])
        img_skip3 = self.model.layer4(img)   # ([B, 2048, 15, 20])
        # img = self.model.avgpool(img_skip3)  # ([B, 2048, 1, 1])
        # img = img.view(img.size(0), -1)# ([B, 2048])
        # # m2 = self.model.fc(img)
        # # m2 = self.model.fc_bn(m2)
        # # m2 = F.relu(m2)
        # m2 = F.relu(self.model.fc_bn(self.model.fc(img))) # ([8, zdim])



        # ----------------------------------Attention Block----------------------------------
        # vectorize the img feature maps
        img_skip1 = img_skip1.view((img_skip1.shape[0], img_skip1.shape[1], -1))   # ([B, 256, 120*160=19200])
        img_skip2 = img_skip2.view((img_skip2.shape[0], img_skip2.shape[1], -1))   # ([B, 512, 60*80=4800])
        img_skip3 = img_skip3.view((img_skip3.shape[0], img_skip3.shape[1], -1))   # ([B, 2048, 15*20=300])

        # Form query, key and value
        # q1 = self.Wq1(x_skip1).transpose(2, 1)   # transpose: ([B, 256, 52000])   --> ([B, 52000, 256])
        # k1 = self.Wk1(img_skip1)                 #            ([B, 256, 19200])  
        # v1 = self.Wv1(img_skip1).transpose(2, 1) # transpose: ([B, 256, 19200])   --> ([B, 19200, 256])

        # q2 = self.Wq2(x_skip2).transpose(2, 1)   # transpose: ([B, 512, 52000])   --> ([B, 52000, 512])
        # k2 = self.Wk2(img_skip2)                 #            ([B, 512, 4800]) 
        # v2 = self.Wv2(img_skip2).transpose(2, 1) # transpose: ([B, 512, 4800])    --> ([B, 4800, 512])

        # q3 = self.Wq3(x_skip3).transpose(2, 1)   # transpose: ([B, 2048, 52000])  --> ([B, 52000, 2048])
        # k3 = self.Wk3(img_skip3)                 #            ([B, 2048, 300])
        # v3 = self.Wv3(img_skip3).transpose(2, 1) # transpose: ([B, 2048, 300])    --> ([B, 300, 2048])

        q1 = x_skip1.transpose(2, 1)   # transpose: ([B, 256, 52000])   --> ([B, 52000, 256])
        k1 = img_skip1                 #            ([B, 256, 19200])  
        v1 = img_skip1.transpose(2, 1) # transpose: ([B, 256, 19200])   --> ([B, 19200, 256])

        q2 = x_skip2.transpose(2, 1)   # transpose: ([B, 512, 52000])   --> ([B, 52000, 512])
        k2 = img_skip2                 #            ([B, 512, 4800]) 
        v2 = img_skip2.transpose(2, 1) # transpose: ([B, 512, 4800])    --> ([B, 4800, 512])

        q3 = x_skip3.transpose(2, 1)   # transpose: ([B, 2048, 52000])  --> ([B, 52000, 2048])
        k3 = img_skip3                 #            ([B, 2048, 300])
        v3 = img_skip3.transpose(2, 1) # transpose: ([B, 2048, 300])    --> ([B, 300, 2048])

        # Calculate W attention weights
        w1 = torch.bmm(q1, k1)   # ([B, 52000, 19200])
        w1 = torch.softmax(w1, dim=1)
        # Calculate the attention values - the fused feature maps
        v_w1 = torch.bmm(w1, v1)   # ([B, 52000, 256])

        # Calculate W attention weights
        w2 = torch.bmm(q2, k2)   # ([B, 52000, 4800])
        w2 = torch.softmax(w2, dim=1)
        # Calculate the attention values - the fused feature maps
        v_w2 = torch.bmm(w2, v2)   # ([B, 52000, 512])

        # Calculate W attention weights
        w3 = torch.bmm(q3, k3)   # ([B, 52000, 300])
        w3 = torch.softmax(w3, dim=1)
        # Calculate the attention values - the fused feature maps
        v_w3 = torch.bmm(w3, v3)   # ([B, 52000, 2048])




        # # ----------------------------------Latent Code - maxpooling the last fused feature map----------------------------------
        # # MLP to combine the two code and make the dimensions equal to zdim
        m = torch.max(v_w3, 1, keepdim=True)[0]       # ([B, 1, 2048])
        m = m.view(-1, 2048)                        # ([B, 2048])
        # save skip connection feature maps for the point branch
        fmap_skips = [v_w1, v_w2, v_w3]
        # set a zero 
        v = 0

        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v, fmap_skips





