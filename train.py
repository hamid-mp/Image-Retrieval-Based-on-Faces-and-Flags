import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision.datasets import ImageFolder
from models import CreateModel
from torchvision import transforms as tfs


BATCH_SIZE = 32
EPOCHS = 100
LR = 0.01

tfs.Compose([
    tfs.RandomCrop((112, 112)),
    tfs.RandomVerticalFlip(0.5),
    tfs.RandomHorizontalFlip(p=0.5),
    tfs.RandomGrayscale(0.3),
    tfs.RandomCrop(),
    tfs.ToTensor(),
    
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CreateModel('resnet18.a1_in1k', 80).load_model()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()








def train(model,
          train_loader,
          valid_loader,
          epochs,
          optimizer,
          criterion,
          device):
    
    model = model.to(device)

    for epoch in tqdm(range(epochs)):

        model.train()
        train_loss = 0
        train_acc = 0

        for batch, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            y_pred = model(x)  

            loss = criterion(y_pred, y)
            
            train_loss += loss
            train_acc += ( torch.eq(y_pred.argmax(dim=1), y).sum().item() / len(y)) * 100

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    train_loss /= len(train_loader)
    train_acc /= len(train_loader)

    with torch.inference_mode():
        valid_loss, valid_acc = 0, 0

        for x, y in valid_loader:

            x, y = x.to(device), y.to(device)

            y_pred = model(x)

            loss = criterion(y_pred, y)

            valid_loss += loss

            valid_acc = (torch.eq(y_pred.argmax(dim=1), y).sum().item() / len(y_pred)) * 100

        valid_acc /= len(valid_loader)
        valid_loss /= len(valid_loader)
    print(f'-------[{epoch+1}| {epochs}] --------')
    print(f"Train Acc: {train_acc} \t Train Loss: {train_loss} \n 
          Valid Acc: {valid_acc} \t Valid Loss: {valid_loss}\n")



