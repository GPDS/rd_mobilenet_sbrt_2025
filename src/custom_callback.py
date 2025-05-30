from keras.api.callbacks import Callback

class True_acc(Callback):
  def __init__(self,data_train,data_test):
    super().__init__()
    self.data_train = data_train
    self.data_test = data_test

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}

    loss_train, acc_train = self.model.evaluate(self.data_train,verbose=0)
    loss_test, acc_test = self.model.evaluate(self.data_test, verbose=0)

    logs['train_loss'] = loss_train
    logs['train_acc'] = acc_train
    logs['val_loss'] = loss_test
    logs['val_acc'] = acc_test

    if acc_train is not None and acc_train >= .98:
      self.model.stop_training = True

    # print(f'\nTrain_loss = {loss_train}; Train_acc = {acc_train}')
    # print(f'\nTest_loss = {loss_test}; Test_acc = {acc_test}')