# import torch
# import torch.nn.functional as F
# from torch import nn


# class PointNetEncoder(nn.Module):
#     def __init__(self, zdim, input_dim=6):
#         super().__init__()
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

#         # Mapping to [c], cmean
#         self.fc1_v = nn.Linear(512, 256)
#         self.fc2_v = nn.Linear(256, 128)
#         self.fc3_v = nn.Linear(128, zdim)
#         self.fc_bn1_v = nn.BatchNorm1d(256)
#         self.fc_bn2_v = nn.BatchNorm1d(128)

#     def forward(self, x):
#         x = x.transpose(1, 2)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.bn4(self.conv4(x))
#         x = torch.max(x, 2, keepdim=True)[0]
#         x = x.view(-1, 512)

#         m = F.relu(self.fc_bn1_m(self.fc1_m(x)))
#         m = F.relu(self.fc_bn2_m(self.fc2_m(m)))
#         m = self.fc3_m(m)
#         v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
#         v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
#         v = self.fc3_v(v)

#         # Returns both mean and logvariance, just ignore the latter in deteministic cases.
#         return m, v


#-------------------------------------------Two-branches Model:ResNet50 + PointNet----------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=6):
        super().__init__()
        # -----------------PointNet encoder-----------------
        self.zdim = zdim
        self.conv1 = nn.Conv1d(input_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)

        # Mapping to [c], cmean
        self.fc1_m = nn.Linear(512, 256)
        self.fc2_m = nn.Linear(256, 128)
        self.fc3_m = nn.Linear(128, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(256)
        self.fc_bn2_m = nn.BatchNorm1d(128)
        self.fc_bn3_m = nn.BatchNorm1d(zdim)

        # Mapping to [c], cmean
        self.fc1_v = nn.Linear(512, 256)
        self.fc2_v = nn.Linear(256, 128)
        self.fc3_v = nn.Linear(128, zdim)
        self.fc_bn1_v = nn.BatchNorm1d(256)
        self.fc_bn2_v = nn.BatchNorm1d(128)
        self.fc_bn3_v = nn.BatchNorm1d(zdim)


        # -----------------ResNet50 encoder-----------------
        self.model = models.resnet50(pretrained=True)
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Linear(2048, 1024)
        self.model.fc_bn = nn.BatchNorm1d(1024)


        # -----------------Fusion MLP-----------------
        # Fusion two latent codes together
        self.fc1_con = torch.nn.Linear(2048,1024)  # convert ResNet50 Latent code from 2048 to 1024

        

    


    def forward(self, x, img):
        # -----------------Shape Latent Code from PointNet-----------------
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))
        m1 = F.relu(self.fc_bn2_m(self.fc2_m(m1)))
        m1 = F.relu(self.fc_bn3_m(self.fc3_m(m1)))
        # m1 = self.fc3_m(m1) # 256 dimension
        v = F.relu(self.fc_bn1_v(self.fc1_v(x)))
        v = F.relu(self.fc_bn2_v(self.fc2_v(v)))
        v = F.relu(self.fc_bn3_m(self.fc3_m(v)))
        # v = self.fc3_v(v)


        # -----------------Image Latent Code from ResNet50-----------------
        img = self.model.conv1(img)
        img = self.model.bn1(img)
        img = self.model.relu(img)
        img = self.model.maxpool(img)
        img = self.model.layer1(img)
        img = self.model.layer2(img)
        img = self.model.layer3(img)
        img = self.model.layer4(img)
        img = self.model.avgpool(img)
        img = img.view(img.size(0), -1)
        # m2 = self.model.fc(img)
        # m2 = self.model.fc_bn(m2)
        # m2 = F.relu(m2)
        m2 = F.relu(self.model.fc_bn(self.model.fc(img)))


        # ----------------------------------Fusion MLP----------------------------------
        # MLP to combine the two code and make the dimensions equal to 256
        m = torch.cat([m1, m2], dim=-1) # (B, 1024+1024)
        m = self.fc1_con(m) # convert ResNet50 Latent code from 2048 to 1024


        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v