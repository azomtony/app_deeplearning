import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import gdown
import time


#You need to implement:

class ResSect(nn.Module):
    def __init__(self, n_filter, n_residual_blocks, beginning_stride):
        '''Initialize the sector by creating layers needed
        n_filter: number of filters in the conv layers in the blocks
        n_residual_blocks: number of blocks in this sector
        beginning_stride: the stride of the first conv of the first block in the sector'''
        super().__init__()
        self.n_filter=n_filter
        if self.n_filter==32:
            self.in_size=32
        else:
            self.in_size=int((self.n_filter/2))
        self.n_residual_blocks=n_residual_blocks
        self.beginning_stride=beginning_stride
        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm2d(n_filter)
        self.RB_conv0 = nn.Conv2d(in_channels=self.in_size, out_channels=self.n_filter, kernel_size=3, stride=self.beginning_stride, padding=1)
        self.RB_conv1 = nn.Conv2d(in_channels=self.n_filter, out_channels=self.n_filter, kernel_size=3, stride=1, padding=1)
        self.convSC = nn.Conv2d(in_channels=self.in_size, out_channels=self.n_filter, kernel_size=1,stride=self.beginning_stride)        


    def forward(self, x):
        '''Implement computation performed in the sector
        x: input tensor
        You should return the result tensor
        '''
        x_res=x
        for i in range(self.n_residual_blocks):
            if i==0:
                if self.beginning_stride==1:
                    x0=x_res
                else:
                    x0=self.convSC(x_res)
                x=self.RB_conv0(x)
                x=self.bn(x)
                x=self.relu(x)
                x=self.RB_conv1(x)
                x=self.bn(x)
                x=torch.add(x0,x)
                x=self.relu(x)
                x_res=x
            else:
                x0=x_res
                x=self.RB_conv1(x_res)
                x=self.bn(x)
                x=self.relu(x)
                x=self.RB_conv1(x)
                x=self.bn(x)
                x=torch.add(x0,x)
                x=self.relu(x)
                x_res=x   

        return x_res             

class ResModel(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()

        self.sect1 = ResSect(32, 3, 1)
        self.sect2 = ResSect(64, 3, 2)
        self.sect3 = ResSect(128, 3, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)

        if pretrained:
            self.load_trained_model()

    def load_trained_model(self):
        '''
        You need to implement this function to:
            1. download the saved pretrained model from your online location
            2. load model from the downloaded model file
            3. I used gdown to download the model from google drive
        '''
        drive_url='https://drive.google.com/uc?id=1XmjfmQsgoMF8J1Svj2vxy87fZroVy_8F'
        fname=f'resmodel_cpu.pth'
        try:

            gdown.download(drive_url,fname,quiet=False)
            self.load_state_dict(torch.load(fname))
            print(f'Pre-trained model loaded successfully')

        except Exception as e:
            print(f'Error loading pre-trained model')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.sect1(x)
        x = self.sect2(x)
        x = self.sect3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def train_loop(trainloader,testloader,model,loss_fn,optimizer,epochs):
    losses=[]
    accuracies=[]
    model.train()
    for epoch in range(epochs):
        start_time=time.time()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch,(X,y) in enumerate(trainloader):
            X,y=X.to(device),y.to(device)
            optimizer.zero_grad()
            pred=model(X)
            loss=loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        epoch_loss=running_loss/len(trainloader)
        accuracy = 100 * correct / total
        losses.append(epoch_loss)
        accuracies.append(accuracy)
        end_time=time.time()
        epoch_time = end_time - start_time
        print(f"Epoch [{epoch + 1}/{epochs}],Time:{epoch_time:.2f} Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
        test_model(testloader,model)
    return losses,accuracies

def test_model(testloader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X,y in testloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f"Accuracy on test set: {100 * correct / total:.2f}%")

def recognize(new_img,device):
    model.eval()
    new_img= new_img.to(device)
    pred=model(new_img.unsqueeze(0))
    prediction=pred.argmax(1)
    id=classes[prediction]

    return id


if __name__=='__main__':
    transform = transforms.Compose([transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU for computation.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU for computation.")
    model = ResModel(pretrained=True).to(device)
    test_model(testloader,model)
    #test_model(trainloader,model)
    '''learning_rate=0.001
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    losses,accuracies=train_loop(trainloader,testloader,model,loss_fn,optimizer,30)


    epochs = range(1, len(losses) + 1)
    epochs,losses,accuracies=np.array(epochs),np.array(losses),np.array(accuracies)
    lcurve=np.vstack((epochs,losses,accuracies))
    np.savetxt('lcurve_cpu_30.out',lcurve,fmt="%f")
    torch.save(model.state_dict(),'resmodel_cpu.pth')
    test_model(testloader,model)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.savefig('Loss_curve_cpu_30.png',dpi=300)

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    ind=30
    new_img = images[ind]
    print(new_img.size())

    pred = recognize(new_img,device)

    print('Neural network recognizes this image as:', pred)
    print('true image label:', classes[labels[ind].item()])'''
