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

    #Get all data first
    path = os.path.dirname(__file__)
    legend = []

    '''explicit'''
    data_dir = os.path.join(path, 'data/explicit.json')

    with open(data_dir, 'r') as file:
        exp = json.load(file)
        legend.append('ARC-EE')

    '''Lanczos'''
    imp = []
    for d in [2,10,20]:
        data_dir = os.path.join(path, f'data/implicit_{d}.json')
        with open(data_dir, 'r') as file:
            imp.append(json.load(file))
            legend.append(f'ARC-IE, {d} dim')

    '''SGD'''
    data_dir = os.path.join(path, 'data/sgd.json')

    with open(data_dir, 'r') as file:
        sgd = json.load(file)
        legend.append('SGD')

    #Now lets plot everything
    loss_fig, loss_ax = plt.subplots(1,1)
    acc_fig, acc_ax = plt.subplots(1,1)

    loss_step = 50

    style = '--'
    props = np.cumsum(np.array(exp['props_list']))
    loss_ax.semilogx(props[::loss_step], exp['train_loss'][::loss_step], ls=style)
    acc_ax.semilogx(props[599::600], exp['validation_acc'], ls=style)

    style = '-'
    for result in imp:
        props = np.cumsum(np.array(result['props_list']))
        loss_ax.semilogx(props[::loss_step], result['train_loss'][::loss_step], ls=style)
        acc_ax.semilogx(props[599::600], result['validation_acc'], ls=style)

    style = '-.'
    props = 2*np.array(range(6001))
    loss_ax.semilogx(props[1::loss_step], sgd['train_loss'][::loss_step], ls=style)
    acc_ax.semilogx(props[599::600], sgd['validation_acc'], ls=style)

    loss_ax.legend(legend)
    loss_ax.set_xlabel('Propagations (log scale)')
    loss_ax.set_ylabel('Cross Entropy Loss')
    loss_ax.set_title('MNIST Training Loss')

    acc_ax.legend(legend)
    acc_ax.set_xlabel('Propagations (log scale)')
    acc_ax.set_ylabel('Classification Accuracy')
    acc_ax.set_title('MNIST Testing Accuracy')

    plt.show()

    if args.save:
        save_dir = os.path.join(path, 'plots/mnist_loss.png')
        loss_fig.savefig(save_dir)
        save_dir = os.path.join(path, 'plots/mnist_acc.png')
        acc_fig.savefig(save_dir)
