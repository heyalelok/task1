import torch
import torch.optim as optim
import torch.nn.functional as F

import random
import os
import argparse
import numpy as np
from model import *
import data as dataloader


parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--batch_size', default=128, type=int, help='train batchsize') 
parser.add_argument('--lr', '--learning_rate', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=150, type=int)
parser.add_argument('--seed', default=123)
parser.add_argument('--num_class', default=10, type=int)

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 120))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_CE(train_loader, model, optimizer,epoch):
    print('\nEpoch: %d' % epoch)
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets, _) in enumerate(train_loader):
        model.train()

        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('100')

        
def get_label(net, test_loader):
    net.eval()
    end_pre = torch.zeros(len(test_loader.dataset))
    n = 0

    with torch.no_grad():
        for _, (inputs) in enumerate(test_loader):
            print('ii')
            inputs = inputs.cuda()
            outputs = net(inputs)
            outputs = torch.argmax(outputs, -1)            
            for b in range(inputs.size(0)):
                end_pre[n] = outputs[b]
                n += 1            
            
    return end_pre


def create_model():
    model = ResNet18(num_classes=args.num_class)
    model = model.cuda()
    return model


loader = dataloader.cifar_dataloader(batch_size=args.batch_size, num_workers=4)

print('| Building net')
net = create_model()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

test_loader = loader.run('test')
warmup_trainloader = loader.run('warmup')


for epoch in range(args.num_epochs):
    adjust_learning_rate(optimizer, epoch)

    train_CE(warmup_trainloader, net, optimizer, epoch)


end_pre = get_label(net, test_loader)
end_pre = np.array(end_pre.cpu())
np.save('label_test.npy', end_pre)
np.save('model.npy', net.state_dict() )



