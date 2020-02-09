from torch.utils.tensorboard import SummaryWriter

class logWriter():
  def __init__(self):
    self.tb = None

  # save for each run params
  def begin_run(self, run):
    self.tb = SummaryWriter(comment=f'-{run}')


  def add_to_board(self, train_loss, test_loss, acc, num_epoch):
    # Record epoch loss and accuracy to TensorBoard
    self.tb.add_scalar('Test Loss', test_loss, num_epoch)
    self.tb.add_scalar('Train Loss', train_loss, num_epoch)
    self.tb.add_scalar('Accuracy', acc, num_epoch)

