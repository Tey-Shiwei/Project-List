from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
from torch.utils import data
from os import makedirs
import torchvision
from PIL import Image
import sys
import copy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def numpy_loader(input):
    item = np.load(input)/255.0
    return Image.fromarray(item)

def evaluate_model_for_accuracy(model, device, data_loader):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print('\n Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))


def evaluate_adv_images(model, device, kwargs, mean, std, data_loader):
    batch_size = 100
    model.eval()

    adv_data_loader = torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('adv_images', #Change this to your adv_images folder
                                           loader=numpy_loader,
                                           extensions='.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         transforms.Normalize(mean, std)])),
                                                                         batch_size=batch_size, **kwargs)

    given_dataset = []
    adv_images = []
    labels = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if len(given_dataset) ==0:
                given_dataset = data.squeeze().detach().cpu().numpy()
            else:
                given_dataset = np.concatenate([given_dataset, data.squeeze().detach().cpu().numpy()],
                                           axis=0)

        for data, target in adv_data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            label = target.squeeze().detach().cpu().numpy()
            softmax_values = torch.nn.Softmax()(output).cpu().numpy()[np.arange(batch_size), label]
            adv_images = data
            labels = target

    #Checking the range of generated images
    adv_images_copy = copy.deepcopy(adv_images)
    for k in range(adv_images_copy.shape[0]):
        image_ = adv_images_copy[k, :, :]

        for t, m, s in zip(image_, mean, std):
            t.mul_(s).add_(m)

        image = image_.squeeze().detach().cpu().numpy()
        image = 255.0 * image

        if np.min(image) < 0 or np.max(image) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

    adv_images = adv_images.squeeze().detach().cpu().numpy()
    labels = labels.squeeze().detach().cpu().numpy()


    #Checking for equation 2 and equation 3
    if all([x > 0.8 for x in softmax_values.tolist()]):
        print('Softmax values for all of your adv images are greater than 0.8')
        S = 0
        for i in range(10):
            label_indices = np.where(labels==i)[0]
            a_i = adv_images[label_indices, :, :]
            for k in range(10):
                image = a_i[k, :, :]
                S = S + np.min(
                            np.sqrt(
                                np.sum(
                                    np.square(
                                        np.subtract(given_dataset, np.tile(np.expand_dims(image, axis=0), [1000,1,1]))
                                    ),axis=(1,2))))

        print('Value of S : {:.4f}'.format(S / 100))

    else:
        print('Softmax values for some of your adv images are less than 0.8')



from attacks import FGSM, BIM, DeepFool
from matplotlib import pyplot as plt

def eq3(image, data_loader):
    device = torch.device("cpu")
    given_dataset = []
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        if len(given_dataset) ==0:
            given_dataset = data.squeeze().detach().cpu().numpy()
        else:
            given_dataset = np.concatenate([given_dataset, data.squeeze().detach().cpu().numpy()],
                                       axis=0)
    given_dataset = given_dataset.reshape(-1,28,28)
    
    S = np.min(np.sqrt(np.sum(
        np.square(np.subtract(given_dataset, np.tile(np.expand_dims(image, axis=0), [1000,1,1]))),axis=(1,2))))
    return S / 100   

def add_image(SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,softmax_clean,softmax_adv,S,label_clean,label_adv,adv_images,image_names,EPS,i):
    if len(S)<10:
        softmax_clean.append(SOFTMAX_CLEAN)
        softmax_adv.append(SOFTMAX_ADV)        
        S.append(S_ADV)
        label_clean.append(np.argmax(LABEL_CLEAN))
        label_adv.append(np.argmax(LABEL_ADV)) ##
        adv_images.append(xadv) ##
        image_names.append("{}_eps{}_{:03}".format(np.argmax(LABEL_ADV),EPS,i)) ##
    else:
         if not(all(s>=S_ADV for s in S)) and all(torch.sum(img-xadv)!=0 for img in adv_images):
            min_index = S.index(min(S))

            del softmax_clean[min_index]
            del softmax_adv[min_index]
            del S[min_index]
            del label_clean[min_index]
            del label_adv[min_index]
            del adv_images[min_index]
            del image_names[min_index]

            softmax_clean.append(SOFTMAX_CLEAN)
            softmax_adv.append(SOFTMAX_ADV)        
            S.append(S_ADV)
            label_clean.append(np.argmax(LABEL_CLEAN))
            label_adv.append(np.argmax(LABEL_ADV)) ##
            adv_images.append(xadv) ##
            image_names.append("{}_eps{}_{:03}".format(np.argmax(LABEL_ADV),EPS,i)) ##
            
    return softmax_clean,softmax_adv,S,label_clean,label_adv,adv_images,image_names

def generate_adv_images(model, device, data_loader):
    
    for batch_idx, (data, target) in enumerate(data_loader):
        xclean = data
        yclean = target
        break
        
    adv_images = []
    image_names = []    
    label_clean = []
    label_adv = []
    softmax_clean = []
    softmax_adv = []
    S = []
    
    adv_images0 = []
    image_names0 = []    
    label_clean0 = []
    label_adv0 = []
    softmax_clean0 = []
    softmax_adv0 = []
    S0 = []

    adv_images1 = []
    image_names1 = []    
    label_clean1 = []
    label_adv1 = []
    softmax_clean1 = []
    softmax_adv1 = []
    S1 = []
    
    adv_images2 = []
    image_names2 = []    
    label_clean2 = []
    label_adv2 = []
    softmax_clean2 = []
    softmax_adv2 = []
    S2 = []
    
    adv_images3 = []
    image_names3 = []    
    label_clean3 = []
    label_adv3 = []
    softmax_clean3 = []
    softmax_adv3 = []
    S3 = []
    
    adv_images4 = []
    image_names4 = []    
    label_clean4 = []
    label_adv4 = []
    softmax_clean4 = []
    softmax_adv4 = []
    S4 = []
    
    adv_images5 = []
    image_names5 = []    
    label_clean5 = []
    label_adv5 = []
    softmax_clean5 = []
    softmax_adv5 = []
    S5 = []
    
    adv_images6 = []
    image_names6 = []    
    label_clean6 = []
    label_adv6 = []
    softmax_clean6 = []
    softmax_adv6 = []
    S6 = []
    
    adv_images7 = []
    image_names7 = []    
    label_clean7 = []
    label_adv7 = []
    softmax_clean7 = []
    softmax_adv7 = []
    S7 = []
    
    adv_images8 = []
    image_names8 = []    
    label_clean8 = []
    label_adv8 = []
    softmax_clean8 = []
    softmax_adv8 = []
    S8 = []
    
    adv_images9 = []
    image_names9 = []    
    label_clean9 = []
    label_adv9 = []
    softmax_clean9 = []
    softmax_adv9 = []
    S9 = []
    
    pmax =  2.82148653034729
    pmin = -0.424212917883804
    
    model.eval()
        
    xadv = torch.zeros(xclean[0].shape)
    
    EPS_ = [20,40,60,80,85,90,100]
    iter_ = 50
    
    for EPS in EPS_:
        for i in range(xclean.shape[0]):
            if i%(xclean.shape[0]/4)==0 and i>0:
                print("Status: {}% for eps={}".format(100*i/xclean.shape[0],EPS))
#             attacker = DeepFool(max_iter=50, clip_max=pmax, clip_min=pmin)
            attacker = BIM(eps=EPS, eps_iter=EPS/iter_, n_iter=iter_, clip_max=pmax, clip_min=pmin)
#             attacker = FGSM(eps=EPS, clip_max=pmax, clip_min=pmin)
            xadv = attacker.generate(model, xclean[i], yclean[i])

            clean = xclean[i][0]
            adv = xadv[0]

            SOFTMAX_CLEAN = np.max(torch.nn.Softmax(dim=1)(model(clean[None,None,:])).detach().cpu().numpy())
            SOFTMAX_ADV = np.max(torch.nn.Softmax(dim=1)(model(adv[None,None,:])).detach().cpu().numpy())
            S_ADV = eq3(adv, data_loader)
            LABEL_CLEAN = model(clean[None,None,:]).squeeze().detach().cpu().numpy()
            LABEL_ADV = model(adv[None,None,:]).squeeze().detach().cpu().numpy()

            if SOFTMAX_ADV>0.8:
                if np.argmax(LABEL_ADV) == 0:
                    softmax_clean0,softmax_adv0,S0,label_clean0,label_adv0,adv_images0,image_names0 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean0,softmax_adv0,S0,label_clean0,label_adv0,adv_images0,image_names0,EPS,i)
                elif np.argmax(LABEL_ADV) == 1:
                    softmax_clean1,softmax_adv1,S1,label_clean1,label_adv1,adv_images1,image_names1 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean1,softmax_adv1,S1,label_clean1,label_adv1,adv_images1,image_names1,EPS,i)
                elif np.argmax(LABEL_ADV) == 2:
                    softmax_clean2,softmax_adv2,S2,label_clean2,label_adv2,adv_images2,image_names2 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean2,softmax_adv2,S2,label_clean2,label_adv2,adv_images2,image_names2,EPS,i)
                elif np.argmax(LABEL_ADV) == 3:
                    softmax_clean3,softmax_adv3,S3,label_clean3,label_adv3,adv_images3,image_names3 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean3,softmax_adv3,S3,label_clean3,label_adv3,adv_images3,image_names3,EPS,i)
                elif np.argmax(LABEL_ADV) == 4:
                    softmax_clean4,softmax_adv4,S4,label_clean4,label_adv4,adv_images4,image_names4 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean4,softmax_adv4,S4,label_clean4,label_adv4,adv_images4,image_names4,EPS,i)
                elif np.argmax(LABEL_ADV) == 5:
                    softmax_clean5,softmax_adv5,S5,label_clean5,label_adv5,adv_images5,image_names5 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean5,softmax_adv5,S5,label_clean5,label_adv5,adv_images5,image_names5,EPS,i)
                elif np.argmax(LABEL_ADV) == 6:
                    softmax_clean6,softmax_adv6,S6,label_clean6,label_adv6,adv_images6,image_names6 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean6,softmax_adv6,S6,label_clean6,label_adv6,adv_images6,image_names6,EPS,i)
                elif np.argmax(LABEL_ADV) == 7:
                    softmax_clean7,softmax_adv7,S7,label_clean7,label_adv7,adv_images7,image_names7 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean7,softmax_adv7,S7,label_clean7,label_adv7,adv_images7,image_names7,EPS,i)
                elif np.argmax(LABEL_ADV) == 8:
                    softmax_clean8,softmax_adv8,S8,label_clean8,label_adv8,adv_images8,image_names8 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean8,softmax_adv8,S8,label_clean8,label_adv8,adv_images8,image_names8,EPS,i)
                elif np.argmax(LABEL_ADV) == 9:
                    softmax_clean9,softmax_adv9,S9,label_clean9,label_adv9,adv_images9,image_names9 = add_image(
                        SOFTMAX_CLEAN,SOFTMAX_ADV,S_ADV,LABEL_CLEAN,LABEL_ADV,xadv,
                        softmax_clean9,softmax_adv9,S9,label_clean9,label_adv9,adv_images9,image_names9,EPS,i)
                
    
    label_adv = label_adv0+label_adv1+label_adv2+label_adv3+label_adv4+label_adv5+label_adv6+label_adv7+label_adv8+label_adv9
    S = S0+S1+S2+S3+S4+S5+S6+S7+S8+S9
    image_names=image_names0+image_names1+image_names2+image_names3+image_names4+image_names5+image_names6+image_names7+image_names8+image_names9
    adv_images=adv_images0+adv_images1+adv_images2+adv_images3+adv_images4+adv_images5+adv_images6+adv_images7+adv_images8+adv_images9
#     print(label_clean)
#     print(softmax_clean)
#     print(label_adv)
#     print(softmax_adv)
#     print("S",S)
#     print("S total: ",sum(S))
    return adv_images,image_names,label_adv

def main():
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-cuda', action='store_true', default=False,help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--model_path', type=str, default='model/mnist_cnn.pt')
    parser.add_argument('--data_folder', type=str, default='data')


    args = parser.parse_args([])
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    mean = (0.1307,)
    std = (0.3081,)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data_loader =  torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('data',
                                           loader= numpy_loader,
                                           extensions= '.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])),
                                           batch_size=args.batch_size, **kwargs)

    model = Net().to(device)

    model.load_state_dict(torch.load(args.model_path, map_location = torch.device('cpu')))

    evaluate_model_for_accuracy(model, device, data_loader)
    
    data_loader2 =  torch.utils.data.DataLoader(
        torchvision.datasets.DatasetFolder('data',
                                           loader= numpy_loader,
                                           extensions= '.npy',
                                           transform=transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize(mean, std)])),
                                           batch_size=1000, **kwargs)
    adv_images,image_names,class_label = generate_adv_images(model, device, data_loader2)
    #Implement this method to generate adv images
    #statisfying constraints mentioned in the assignment discription

    save_folder = 'adv_images'

    for image,image_name,class_label in zip(adv_images,image_names,class_labels):
        for t, m, s in zip(image, mean, std):
            t.mul_(s).add_(m)

        image_to_save = image.squeeze().detach().cpu().numpy()
        image_to_save = 255.0 * image_to_save

        if np.min(image_to_save) < 0 or np.max(image_to_save) > 255:
            print('Generated adversarial image is out of range.')
            sys.exit()

        if not os.path.exists(os.path.join(save_folder,str(class_label))):
            makedirs(os.path.join(save_folder,str(class_label)))

        np.save(os.path.join(save_folder,str(class_label),image_name), image_to_save)

    evaluate_adv_images(model,device,kwargs,mean,std,data_loader)


if __name__ == '__main__':
    main()
