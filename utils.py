import torch
import matplotlib.pyplot as plt
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

class EarlyStopper():

    def __init__(self, model, min_delta, patience):
        self.min_delta = min_delta
        self.patience = patience
        self.min_validation_loss = float('inf')
        self.counter = 0
        self.model = model
    def early_stop(self, validation_loss):

        if self.min_validation_loss > validation_loss: # loss decreased
            self.save_model(self.model)
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (self.min_validation_loss + self.min_delta) < validation_loss: #if (previous loss + a margin ) is'nt bigger than current loss => loss doesnt decrease well
            self.counter += 1   

            if self.counter > self.patience:
                return True # if True => break
        return False
    
    @staticmethod
    def save_model(model):
        torch.save(model.state_dict(), f='./weights/best.pt')
        print('Saving model...')



def confusion_matrix(class_names, y_pred_tensor, targets):

    # 2. Setup confusion matrix instance and compare predictions to targets
    if isinstance(class_names, int):
        class_names = list(range(class_names))


        confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')

    confmat_tensor = confmat(preds=y_pred_tensor,
                            target=targets)

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
        class_names=class_names, # turn the row and column labels into class names
        figsize=(10, 7)
    );
    fig.savefig('ConfusionMatrix.png')


def training_plots(vloss, vacc, tloss, tacc):
    vloss = torch.tensor(vloss, device = 'cpu')
    vacc = torch.tensor(vacc, device = 'cpu')
    tloss = torch.tensor(tloss, device = 'cpu')
    tacc = torch.tensor(tacc, device = 'cpu')

    plt.plot(tloss, label="train")
    plt.plot(vloss, label="test")
    plt.xlabel("epochs")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.show()
    plt.savefig('./Training Loss.png')

    plt.plot(tacc, label="train")
    plt.plot(vacc, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()  
    plt.savefig('./Training Accuracy.png')