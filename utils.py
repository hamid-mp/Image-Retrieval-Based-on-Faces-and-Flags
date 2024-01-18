import torch
import timm

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




    


