import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
 
device = ('cuda' if torch.cuda.is_available() else 'cpu')
 
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
 
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
 
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=True, download=True,
                   transform=transforms.ToTensor()), batch_size=64, shuffle=True)
 
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='./data', train=False, download=True,
                   transform=transforms.ToTensor()), batch_size=64, shuffle=True)
 
model = LeNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
 
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
 
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
 
def test():
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
 
            # 累加批量损失
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # 获取最大对数概率的索引
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
 
        test_loss /= len(test_loader.dataset)
        print('\nTest set:Average loss: {:.4f}, Accuracy:{}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
 
for epoch in range(1,10+1 ):
    train(epoch)
    test()
 
save_dir = "C:/Users/user/Desktop/code/adversial_attack/FGSM/"
model_name = 'LeNet'
PATH = os.path.join(save_dir, model_name)
torch.save(model.state_dict(), PATH)

