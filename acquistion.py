# Torch
import torch
import numpy as np
from scipy.stats import mode
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from scipy.stats import entropy
## Coreset
from scipy.spatial import distance_matrix

intermediate = {}

def get_intermediate(name):
    def hook(model, input, output):
        #print("lennnn :", len(output))
        intermediate[name] = output.detach()
        #for idx, o  in enumerate(output):
        #    print("idx: ",idx)
        #    intermediate[name+str(idx)] = o.detach()
        #intermediate[name] = output.detach()
    return hook


def Coreset(device, model, labeled_loader, unlabel_loader, amount):
    feature1_hook = model.model._modules["17"].register_forward_hook(get_intermediate("first_fm"))
    feature2_hook = model.model._modules["20"].register_forward_hook(get_intermediate("second_fm"))
    feature3_hook = model.model._modules["23"].register_forward_hook(get_intermediate("third_fm"))
    avgpool = nn.AdaptiveAvgPool2d((1,1))

    model.eval()
    unlabeled_representation = torch.tensor([]).cuda()
    labeled_representation = torch.tensor([]).cuda()
    with torch.no_grad():
        for i, (imgs, targets, paths, _) in enumerate(unlabel_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            targets = targets.to(device)
            pred = model(imgs)

            first_repre = avgpool(intermediate['first_fm'])
            second_repre = avgpool(intermediate['second_fm'])
            third_repre = avgpool(intermediate['third_fm'])
            first_repre = first_repre.view(first_repre.size(0), -1)
            second_repre = second_repre.view(second_repre.size(0), -1)
            third_repre = third_repre.view(third_repre.size(0), -1)

            #print("first_repre shape: ", first_repre.shape)
            #print("second_repre shape: ", second_repre.shape)
            #print("third_repre shape: ", third_repre.shape)
            representation = torch.cat((first_repre, second_repre, third_repre), 1)
            #print("representation shape: ", representation.shape)

            unlabeled_representation = torch.cat((unlabeled_representation, representation + 1e-10), 0) # 10000 512

        for i, (imgs, targets, paths, _) in enumerate(labeled_loader):
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
            targets = targets.to(device)
            pred = model(imgs)

            first_repre = avgpool(intermediate['first_fm'])
            second_repre = avgpool(intermediate['second_fm'])
            third_repre = avgpool(intermediate['third_fm'])
            first_repre = first_repre.view(first_repre.size(0), -1)
            second_repre = second_repre.view(second_repre.size(0), -1)
            third_repre = third_repre.view(third_repre.size(0), -1)
            representation = torch.cat((first_repre, second_repre, third_repre), 1)

            labeled_representation = torch.cat((labeled_representation, representation + 1e-10), 0) # 10000 512
            
            
            
            
            
        

        #print("labeled_representation: ", labeled_representation.shape)
        #print("unlabeled_representation: ", unlabeled_representation.shape)
        new_indices = greedy_k_center(labeled_representation, unlabeled_representation, amount) 
        # return a list
    return new_indices

def greedy_k_center(labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        labeled = labeled.cpu()
        unlabeled = unlabeled.cpu()
        #print("labeled: ", labeled.shape)
        #print("labeled[0, :].reshape((1, labeled.shape[1])): ", labeled[0, :].reshape((1, labeled.shape[1])).shape)
        #print("unlabeled: ", unlabeled.shape)
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return greedy_indices
