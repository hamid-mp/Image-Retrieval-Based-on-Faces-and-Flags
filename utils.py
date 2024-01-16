


class EarlyStopper():

    def __init__(self, min_delta, patience):
        self.min_delta = min_delta
        self.patience = self.patience
        self.min_validation_loss = float('inf')
        self.counter = 0

    def early_stop(self, validation_loss):

        if self.min_validation_loss > validation_loss: # loss decreased
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (self.min_validation_loss + self.min_delta) < validation_loss: #if (previous loss + a margin ) is'nt bigger than current loss => loss doesnt decrease well
            self.counter += 1   

            if self.counter > self.patience:
                return True # if True => break
        return False