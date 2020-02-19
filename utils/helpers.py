# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import plot_confusion_matrix
#
# import torch
# import torch
# from torchvision.transforms import transforms
# from config import get_config
# import numpy as np
# from PIL import Image
#
# from dataloader import dataLoader
# from models import CNNModel3
#
# model = CNNModel3()
# state_dict = torch.load('checkpoints/tt.pt')
# # load dict into the network
# model.load_state_dict(state_dict)
# config = get_config()
# model.eval()
# dataloader = dataLoader(config)
# dataset, train_sampler, valid_sampler = dataloader.getLoaders()
# validation_loader = torch.utils.data.DataLoader(dataset,
#                                                       batch_size=256,
#                                                       sampler=valid_sampler)
#
# nb_classes = 9
#
# # Initialize the prediction and label lists(tensors)
# predlist=torch.zeros(0,dtype=torch.long, device='cpu')
# lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
#
# with torch.no_grad():
#     for i, (inputs, classes) in enumerate(validation_loader):
#
#         outputs = model(inputs)
#         _, preds = torch.max(outputs, 1)
#
#         # Append batch prediction results
#         predlist=torch.cat([predlist,preds.view(-1).cpu()])
#         lbllist=torch.cat([lbllist,classes.view(-1).cpu()])
#
# # Confusion matrix
# conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
# print(conf_mat)
#
# # Per-class accuracy
# class_accuracy=100*conf_mat.diagonal()/conf_mat.sum(1)
# print(class_accuracy)

def reset_model_weights(model):
    for name, module in model.named_children():
        if name not in ['pool', 'dropout']:
            module.reset_parameters()
def check_triplet(model):
    for name, module in model.named_children():
        if name in ['TripletNet']:
            return True
    return False
