'''
Casting the logistic regression problem as an MLP neural
network using PyTorch.
'''
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

'''
Build model, train, and validate -- potentially using second order optimizer.
'''
def spambase(dataroot, model_type='zero', optim_method=None, order=1, batch_size=64,
                epochs=1, learn_rate=0.01, **kw):

    #Check for GPU
    # if torch.cuda.is_available():
    #     device = 'cuda:0'
    #     print('Using GPU acceleration.')
    # else:
    #     device = 'cpu'
    #     print('Using CPU only.')

    #Only gonna use cpu for now
    device = torch.device('cpu')

    input_dim = 57

    #Load the model, setup loss and optimizer
    if model_type=='zero':
        model=ZeroHidden(input_dim)
    elif model_type=='one':
        model = OneHidden(input_dim)
    else:
        print('Model not supported.')
        quit()

    model.to(device)
    loss = torch.nn.BCEWithLogitsLoss()

    if optim_method is not None and order == 2:
        optimizer = optim_method(model.parameters(), **kw)
    elif optim_method is not None:
        optimizer = optim_method(model.parameters(), lr=learn_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    #Setup dataloader
    train = SpamBase(dataroot, 'train', device=device)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)

    #Now train the model
    print('Starting to train...')

    for epoch in range(epochs):
        for sample in trainloader:
            optimizer.zero_grad()

            data = sample['features']
            labels = sample['labels']

            def loss_fn():
                with torch.no_grad():
                    outputs = model(data)
                    loss_val = loss(outputs, labels)

                    return loss_val

            if order == 2:
                '''
                There is probably a cleaner way to do this that is also more
                reusable, but this accomplishes the goal.

                I could also probably turn the gradients into a single vector
                here if that is the route I am going down.
                '''
                outputs = model(data)
                loss_val = loss(outputs, labels)

                #First get full gradient
                grads = torch.autograd.grad(loss_val, model.parameters())
                grads = [g.detach().data for g in grads]

                #Then get sub sample for hessian
                idx = torch.randint(low=1, high=data.shape[0],
                                        size=(int(np.ceil(batch_size*sample_rate)),))

                gradsH = torch.autograd.grad(loss(model(data[idx,...]), labels[idx,...]),
                                                model.parameters(), create_graph=True)

                optimizer.step(grads, gradsH, loss_val, loss_fn)

            else:
                model.zero_grad()
                outputs = model(data)
                loss_val = loss(outputs, labels)
                loss_val.backward()
                optimizer.step()

    print('Training finished.')

    #Validate
    test = SpamBase(dataroot, 'test', device=device)
    testloader = DataLoader(test, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    correct, total = 0, 0
    for features, labels in testloader:
        predictions = model.predict(data['features'])

        total += data['labels'].size(0)
        correct += (predictions == data['labels']).sum()

    print(f'Test Accuracy: {(correct/total).item()}')

if __name__=='__main__':
    spambase(dataroot='./spambase')
