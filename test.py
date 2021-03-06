import numpy as np
from matplotlib import pyplot as plt
from torchvision import datasets, transforms

import imageio
from skimage.transform import resize
from skimage.color import rgb2gray

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import network as net

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

train_set = datasets.MNIST('DATA_MNIST/', download=True, train=True, transform=transform)
trainLoader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

training_data = enumerate(trainLoader)
batch_idx, (images, labels) = next(training_data)

model = net.Network()

model = torch.load('./model.pt')
model.eval()

i = 0
while (1):
    #source_img = Image.open("test_img.png")
    #img_tensor = transform(source_img)
    #img_tensor = img_tensor.unsqueeze(0)
    #img_tensor = img_tensor.view(-1, 1, 28, 28)

    #print(img_tensor)

    img = images[i]
    img = img.view(-1, 1, 28, 28)

    with torch.no_grad():
        logits = model.forward(img)

    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,8), ncols=2)
    ax1.imshow(img.view(1, 28, 28).detach().cpu().numpy().squeeze(), cmap='inferno')
    ax1.axis('off')
    ax2.barh(np.arange(10), probabilities, color='r' )
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.show()
    i += 1
