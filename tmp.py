import torch
from models import Darknet

net = Darknet('./cfg/yolov4', (602, 602))
x = torch.zeros([1, 3, 602, 602])

pre = net(x)

for i in pre:
    print(i.shape)
