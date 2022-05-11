'''
Image classification tasks using PyTorch with support for second order optimization
in the form of SARC. Currently we support MNIST hand-written digits and CIFAR10
'''
from core.optimizer import SARC
from core.models.models import Dense3, CNN2

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import time
import os

def evaluate(model, loader):
    with torch.no_grad():
        correct, total = 0, 0
        model.eval()
        for data, labels in loader:
            predictions = model.predict(data)

            total += labels.size(0)
            correct += (predictions == labels).sum()

        return (correct/total)

'''
Build model, train, and validate -- potentially using second order optimizer
'''
def imageClassification(dataset, optim_method, state=None, epochs=1, batch_size=64,
                            sample_rate=0.05, return_model=False, validate=None, **kw):
    #Setup dataloaders
    if dataset == 'mnist':
        model_type = Dense3
        DS = datasets.MNIST
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])

    elif dataset == 'cifar10':
        model_type = CNN2
        DS = datasets.CIFAR10
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                (0.247, 0.243, 0.261))])

    else:
        raise ValueError('Requested dataset not supported.')

    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, 'data/' + dataset)

    #Training data
    trainset = DS(data_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)

    #Testing data
    testset = DS(data_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)

    #Validation data
    if validate is not None:
        assert validate <= 1 and validate >= 0
        length = int(validate*len(testset))
        valset, _ = random_split(testset, [length, len(testset)-length])
        valloader = DataLoader(valset, batch_size=batch_size, shuffle=True,
                                    num_workers=4, pin_memory=True)

    #Build model
    model = model_type()

    if state != None:
        print('Loading model state dict')
        model.load_state_dict(state)

    device = torch.device('cpu')
    model.to(device)

    #Loss function
    loss = torch.nn.CrossEntropyLoss()

    #Optimizer
    if optim_method == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **kw)
        order = 1
    elif optim_method == 'sarc':
        optimizer = SARC(model.parameters(), **kw)
        order = 2
    else:
        raise ValueError('Only SGD and SARC supported as optimizers.')


    #Setup performance recording
    val_acc = []
    train_loss = []
    step_time, step_calls = 0.0, 0

    #Train
    model.train()

    print('Starting to train...')
    tic = time.perf_counter()

    for epoch in range(epochs):
        for i, (data, labels) in enumerate(trainloader):

            def loss_fn():
                with torch.no_grad():
                    outputs = model(data)
                    loss_val = loss(outputs, labels)

                    return loss_val

            if order == 2:
                outputs = model(data)
                loss_val = loss(outputs, labels)

                '''I could do this portion slightly better by doing the gradsH
                first and then doing grads on only the data not included in gradH
                but then adding the gradsH value to grads, i.e. only ever do a
                forward pass on a piece of data once.'''

                #First get full gradient
                grads = torch.autograd.grad(loss_val, model.parameters())
                grads = [g.detach().data for g in grads]

                #Then get sub sample for hessian
                idx = torch.randint(low=1, high=data.shape[0],
                                        size=(int(np.ceil(batch_size*sample_rate)),))

                gradsH = torch.autograd.grad(loss(model(data[idx,...]), labels[idx,...]),
                                                model.parameters(), create_graph=True)

                step_calls += 1
                s_tic = time.perf_counter()

                optimizer.step(grads, gradsH, loss_val.data, loss_fn)

                step_time += time.perf_counter() - s_tic

            else:
                model.zero_grad()
                outputs = model(data)
                loss_val = loss(outputs, labels)
                loss_val.backward()

                step_calls += 1
                s_tic = time.perf_counter()

                optimizer.step()

                step_time += time.perf_counter() - s_tic

            #Save training loss info
            train_loss.append(loss_val.item())

        #Get validation accuracy every epoch
        if validate:
            acc = evaluate(model, valloader)
            val_acc.append(acc.item())

        #Print info at end of epoch
        print(f'Epoch {epoch}: Loss = {loss_val}, Avg. Step Time = {step_time/step_calls}')

    train_time = time.perf_counter() - tic

    print('Training finished.')

    #Get optimizer training information
    if order == 2:
        info = optimizer.getInfo()
    else:
        info = {}

    #Validate
    acc = evaluate(model, testloader)

    info['test_acc'] = acc.item()
    info['train_time'] = train_time
    info['avg_step_time'] = step_time/step_calls
    info['train_loss'] = train_loss

    if validate:
        info['validation_acc'] = val_acc

    if return_model:
        return info, model
    else:
        return info
