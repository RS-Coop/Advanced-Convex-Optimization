'''
MNIST hand-written digits classification task using Pytorch
'''
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import time
import matplotlib.pyplot as plt

'''
Build model, train, and validate -- potentially using second order optimizer
'''
def mnist(data_dir, optim_method=None, batch_size=128, epochs=1, learn_rate=0.01,
            order=1, sample_rate=0.03, **kw):

    #Check for GPU
    # if torch.cuda.is_available():
    #     device = 'cuda:0'
    #     print('Using GPU acceleration.')
    # else:
    #     device = 'cpu'
    #     print('Using CPU only.')

    #Only gonna use cpu for now
    device = torch.device('cpu')

    model = Dense()

    model.to(device)
    loss = torch.nn.CrossEntropyLoss()

    if optim_method is not None and order == 2:
        optimizer = optim_method(model.parameters(), **kw)
    elif optim_method is not None:
        optimizer = optim_method(model.parameters(), lr=learn_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    #Setup dataloader
    transform = transforms.Compose([transforms.ToTensor()])

    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    trainloader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)

    #Train
    model.train()
    print('Starting to train...')
    for epoch in range(epochs):
        e_tic = time.perf_counter()
        for i, (data, labels) in enumerate(trainloader):
            # data = data.to(device)
            # labels = labels.to(device)
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

                s_tic = time.perf_counter()
                optimizer.step(grads, gradsH, loss_val.data, loss_fn)
                s_toc = time.perf_counter()

            else:
                model.zero_grad()
                outputs = model(data)
                loss_val = loss(outputs, labels)
                loss_val.backward()

                s_tic = time.perf_counter()
                optimizer.step()
                s_toc = time.perf_counter()

            '''
            Print batch specific details: step runtime,
            '''
            if i%50 == 0:
                print(f'Epoch {epoch}, Batch {i}: Loss = {loss_val}, Step Time = {s_toc-s_tic}')

        e_toc = time.perf_counter()
        '''
        Print epoch specifc details
        '''
        print(f'\nEpoch {epoch}: Total Time = {e_toc-e_tic}')

    print('Training finished.')

    '''
    Print optimzer training details
    '''
    if order==2:
        info = optimizer.getInfo()
        print(f'\nOptimizer timing: HVP {info[0]}, Scipy Minimize {info[1]}, Sub-Problem Time {info[2]}')
        # plt.plot(range(len(info[3])), info[3])
        # plt.show()


    #Validate
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    testloader = DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)

    correct, total = 0, 0
    model.eval()
    for data, labels in testloader:
        predictions = model.predict(data)

        total += labels.size(0)
        correct += (predictions == labels).sum()

    print(f'Test Accuracy: {(correct/total).item()}')
