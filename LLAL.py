import torch
import torch.nn as nn
import torch.nn.functional as F

class Lossnet(nn.Module):
    def __init__(self):
        super(Lossnet, self).__init__()
        
        num_channels=[128, 256, 512]
        self.FC1 = nn.Linear(128, 128)
        self.FC2 = nn.Linear(256, 128)
        self.FC3 = nn.Linear(512, 128)
        
        self.loss_pred1 = nn.Linear(len(num_channels)*128,128)
        self.loss_pred2 = nn.Linear(128, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.conv1 = nn.Conv2d(128,128,4,1,0)
        self.conv2 = nn.Conv2d(256,256,3,1,0)
        self.conv3 = nn.Conv2d(512,512,2,1,0)

    def forward(self, features):
        '''        
        out1 = self.avgpool(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.avgpool(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.avgpool(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out = self.loss_pred(torch.cat((out1, out2, out3), 1))
    
        '''
        out1 = self.avgpool(self.conv1(features[0]))
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.avgpool(self.conv2(features[1]))
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.avgpool(self.conv3(features[2]))
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out = self.loss_pred2(self.loss_pred1(torch.cat((out1, out2, out3), 1)))



        return out

        '''
        fc_outputs = []
        for idx, feature in enumerate(features):
            out = self.avgpool(feature)
            out = torch.flatten(out, 1)
            out = self.fc[idx](out)
            fc_outputs.append(F.relu(out))
        fc_cat = torch.cat(fc_outputs, dim=1)
        loss_pred = self.loss_pred(fc_cat)
        return out_p, loss_pred
        '''
