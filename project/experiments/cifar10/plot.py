import matplotlib.pyplot as plt
import json
import numpy as np
import os
import argparse
import seaborn as sns
sns.set()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', type=bool, default=False)
    args = parser.parse_args()

    #Data path
    path = os.path.dirname(__file__)

    #Three ablation plots: SGD momentum, SGD lr, ARC-EE sub-problem tol

    loss_step = 50

    #SGD lr
    loss_fig, loss_ax = plt.subplots(1,1)
    acc_fig, acc_ax = plt.subplots(1,1)
    legend = []

    for lr in ['005', '001', '0005', '0001']:
        data_dir = os.path.join(path, f'data/sgd_{lr}_09.json')
        with open(data_dir, 'r') as file:
            data = json.load(file)
            legend.append(f'SGD, lr={lr[0]+"."+lr[1:]}')

        props = 2*np.array(range(12500))
        log_sample = np.rint(np.logspace(0,np.log10(len(props)-1),100)).astype(np.int)
        loss_ax.semilogx(props[log_sample], np.array(data['train_loss'])[log_sample])
        acc_ax.semilogx(props[499::500], data['validation_acc'])

    loss_ax.legend(legend)
    loss_ax.set_xlabel('Propagations (log scale)')
    loss_ax.set_ylabel('Cross Entropy Loss')
    loss_ax.set_title('CIFAR10 Training Loss')

    acc_ax.legend(legend)
    acc_ax.set_xlabel('Propagations (log scale)')
    acc_ax.set_ylabel('Classification Accuracy')
    acc_ax.set_title('CIFAR10 Testing Accuracy')

    plt.show()

    if args.save:
        save_dir = os.path.join(path,'plots/cifar10_sgd_lr_loss.png')
        loss_fig.savefig(save_dir)
        save_dir = os.path.join(path,'plots/cifar10_sgd_lr_acc.png')
        acc_fig.savefig(save_dir)

    #SGD momentum
    loss_fig, loss_ax = plt.subplots(1,1)
    acc_fig, acc_ax = plt.subplots(1,1)
    legend = []

    for m in ['00', '045', '09']:
        data_dir = os.path.join(path, f'data/sgd_001_{m}.json')
        with open(data_dir, 'r') as file:
            data = json.load(file)
            legend.append(f'SGD, momentum={m[0]+"."+m[1:]}')

        props = 2*np.array(range(12500))
        log_sample = np.rint(np.logspace(0,np.log10(len(props)-1),100)).astype(np.int)
        loss_ax.semilogx(props[log_sample], np.array(data['train_loss'])[log_sample])
        acc_ax.semilogx(props[499::500], data['validation_acc'])

    loss_ax.legend(legend)
    loss_ax.set_xlabel('Propagations (log scale)')
    loss_ax.set_ylabel('Cross Entropy Loss')
    loss_ax.set_title('CIFAR10 Training Loss')

    acc_ax.legend(legend)
    acc_ax.set_xlabel('Propagations (log scale)')
    acc_ax.set_ylabel('Classification Accuracy')
    acc_ax.set_title('CIFAR10 Testing Accuracy')

    plt.show()

    if args.save:
        save_dir = os.path.join(path, 'plots/cifar10_sgd_m_loss.png')
        loss_fig.savefig(save_dir)
        save_dir = os.path.join(path, 'plots/cifar10_sgd_m_acc.png')
        acc_fig.savefig(save_dir)

    #ARC-EE tol
    loss_fig, loss_ax = plt.subplots(1,1)
    acc_fig, acc_ax = plt.subplots(1,1)
    legend = []

    for tol in ['01', '001', '0001', '00001']:
        data_dir = os.path.join(path, f'data/explicit_{tol}.json')
        with open(data_dir, 'r') as file:
            data = json.load(file)
            legend.append(f'ARC-EE, tolerance={tol[0]+"."+tol[1:]}')

        props = np.cumsum(np.array(data['props_list']))
        log_sample = np.rint(np.logspace(0,np.log10(len(props)-1),100)).astype(np.int)

        loss_ax.semilogx(props[log_sample], np.array(data['train_loss'])[log_sample])
        acc_ax.semilogx(props[499::500], data['validation_acc'])

    loss_ax.legend(legend)
    loss_ax.set_xlabel('Propagations (log scale)')
    loss_ax.set_ylabel('Cross Entropy Loss')
    loss_ax.set_title('CIFAR10 Training Loss')

    acc_ax.legend(legend)
    acc_ax.set_xlabel('Propagations (log scale)')
    acc_ax.set_ylabel('Classification Accuracy')
    acc_ax.set_title('CIFAR10 Testing Accuracy')

    plt.show()

    if args.save:
        save_dir = os.path.join(path, 'plots/cifar10_arc_tol_loss.png')
        loss_fig.savefig(save_dir)
        save_dir = os.path.join(path, 'plots/cifar10_arc_tol_acc.png')
        acc_fig.savefig(save_dir)

    #Plot sgd 001 09 and arc-ee 0001 together
    loss_fig, loss_ax = plt.subplots(1,1)
    acc_fig, acc_ax = plt.subplots(1,1)
    legend = []

    data_dir = os.path.join(path, f'data/sgd_001_09.json')
    with open(data_dir, 'r') as file:
        data = json.load(file)
        legend.append('SGD')

    props = 2*np.array(range(12500))
    log_sample = np.rint(np.logspace(0,np.log10(len(props)-1),100)).astype(np.int)
    loss_ax.semilogx(props[log_sample], np.array(data['train_loss'])[log_sample])
    acc_ax.semilogx(props[499::500], data['validation_acc'])

    data_dir = os.path.join(path, f'data/explicit_0001.json')
    with open(data_dir, 'r') as file:
        data = json.load(file)
        legend.append('ARC-EE')

    props = np.cumsum(np.array(data['props_list']))
    log_sample = np.rint(np.logspace(0,np.log10(len(props)-1),100)).astype(np.int)
    loss_ax.semilogx(props[log_sample], np.array(data['train_loss'])[log_sample])
    acc_ax.semilogx(props[499::500], data['validation_acc'])

    loss_ax.legend(legend)
    loss_ax.set_xlabel('Propagations (log scale)')
    loss_ax.set_ylabel('Cross Entropy Loss')
    loss_ax.set_title('CIFAR10 Training Loss')

    acc_ax.legend(legend)
    acc_ax.set_xlabel('Propagations (log scale)')
    acc_ax.set_ylabel('Classification Accuracy')
    acc_ax.set_title('CIFAR10 Testing Accuracy')

    plt.show()

    if args.save:
        save_dir = os.path.join(path, 'plots/cifar10_sgd_arc_loss.png')
        loss_fig.savefig(save_dir)
        save_dir = os.path.join(path, 'plots/cifar10_sgd_arc_acc.png')
        acc_fig.savefig(save_dir)
