#-------------------------------------------1st Version - Large Latent Dimension - Two-branches Model:ResNet50 + PointNet----------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models
from .pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class PointNetEncoder(nn.Module):
    def __init__(self, zdim, input_dim=6, normal_channel=True):
        super().__init__()
        # ----------------- PointNet++ encoder -----------------
        self.zdim = zdim
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # # With MSG approch 
        # self.sa1 = PointNetSetAbstractionMsg(4800, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        # self.sa2 = PointNetSetAbstractionMsg(300, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        # self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 2048], True)
        
        # Simplest implementation
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, 6 + 3, [32, 64, 256], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 256 + 3, [256, 256, 512], False)
        self.sa3 = PointNetSetAbstraction(16, 0.8, 32, 512 + 3, [512, 512, 2048], False)
        self.fc1_m = nn.Linear(2048, zdim)
        self.fc_bn1_m = nn.BatchNorm1d(zdim)


        # ----------------- ResNet50 encoder -----------------
        self.model = models.resnet50(pretrained=True)
        #self.model.load_state_dict(torch.load('./model/resnet50-19c8e357.pth')) # Turn pretrained to False then load the parameters locally to save time
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.fc = nn.Linear(2048, zdim)
        self.model.fc_bn = nn.BatchNorm1d(zdim)


        # ----------------- Fusion MLP -----------------
        # Fusion two latent codes together
        self.fc1_con = torch.nn.Linear(2*zdim, zdim)  # convert ResNet50 Latent code from 2048 to 1024

        

    


    def forward(self, x, img):
        # ----------------- Shape Latent Code from PointNet++ -----------------
                                                   # ([B, 52000, 6])
        x = x.transpose(1, 2)                      # ([B, 6, 52000])

        l0_points = x
        l0_xyz = x[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points = torch.max(l3_points, 2, keepdim=True)[0]
        x = l3_points.view(-1, 2048)                        # ([B, z_dim])
        
        # m1 = x                                              # ([B, z_dim])
        m1 = F.relu(self.fc_bn1_m(self.fc1_m(x)))  # ([B, z_dim])



        # # -----------------Image Latent Code from ResNet50-----------------
        # img = self.model.conv1(img)    # ([8, 64, 240, 320])
        # img = self.model.bn1(img)      # ([8, 64, 240, 320])
        # img = self.model.relu(img)     # ([8, 64, 240, 320])
        # img = self.model.maxpool(img)  # ([8, 64, 120, 160])
        # img = self.model.layer1(img)   # ([8, 256, 120, 160])
        # img = self.model.layer2(img)   # ([8, 512, 60, 80])
        # img = self.model.layer3(img)   # ([8, 1024, 30, 40])
        # img = self.model.layer4(img)   # ([8, 2048, 15, 20])
        # img = self.model.avgpool(img)  # ([8, 2048, 1, 1])
        # img = img.view(img.size(0), -1)# ([8, 2048])
        # # m2 = self.model.fc(img)
        # # m2 = self.model.fc_bn(m2)
        # # m2 = F.relu(m2)
        # m2 = F.relu(self.model.fc_bn(self.model.fc(img))) # ([8, 64])


        # ----------------------------------Fusion MLP----------------------------------
        # # MLP to combine the two code and make the dimensions equal to zdim
        # m = torch.cat([m1, m2], dim=-1) # (B, zdim+zdim)
        # m = self.fc1_con(m) # convert ResNet50 Latent code from 2*zdim to zdim
        m = m1

        # v = torch.cat([v, m2], dim=-1) # (B, zdim+zdim)
        # v = self.fc1_con(v) # convert ResNet50 Latent code from 2*zdim to zdim
        v = 0


        # Returns both mean and logvariance, just ignore the latter in deteministic cases.
        return m, v
    







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
