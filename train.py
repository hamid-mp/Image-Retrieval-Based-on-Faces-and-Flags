import torch
from pathlib import Path
from utils import EarlyStopper, save_model
from tqdm import tqdm
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models import CreateModel
from torchvision import transforms as tfs
from loss import FocalLoss

# To do:
# 1- Save Model on performance
# 2- Add Stopping Criteria : Done
# 3- Step LR

(Path(__file__).parent.resolve() / 'weights' ).mkdir(parents=True, exist_ok=True)
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.01

transforms = tfs.Compose([
    tfs.RandomVerticalFlip(0.5),
    tfs.RandomHorizontalFlip(p=0.5),
    tfs.RandomGrayscale(0.3),
    tfs.Resize((224,224)),
    tfs.ToTensor(),
    
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MANAGER = CreateModel('resnet18.a1_in1k', 84)
model = MANAGER.build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = FocalLoss()

stopper = EarlyStopper(model=model, min_delta=0.02, patience=5)


train_set = ImageFolder(root='./FlagCrops', transform=transforms)
train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
def train(model,
          train_loader,
          valid_loader,
          epochs,
          optimizer,
          criterion,
          device, stop_criteria):
    
    model = model.to(device)
    
    for epoch in tqdm(range(epochs)):
        print(f'-------[{epoch+1}|{epochs}] --------')

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

        print(f"Train Acc: {train_acc} \t Train Loss: {train_loss}")
        if valid_loader:
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
                print(f"Valid Acc: {valid_acc} \t Valid Loss: {valid_loss}\n")

        if stop_criteria.early_stop(valid_loss):
            print("Training has been stopped due to early stopping criteria")
            break



if __name__ == '__main__':
    train(model,
          train_loader,
          None,
          EPOCHS,
          optimizer,
          criterion,
          device,
          stopper)


