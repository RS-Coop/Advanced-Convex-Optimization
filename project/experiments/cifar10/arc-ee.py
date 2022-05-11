from core.problems import imageClassification as imc
import json
import torch
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='CIFAR10 explicit sub-problem test')
    parser.add_argument('--tol', type=float, default=1e-2)
    parser.add_argument('--save_model', type=bool, default=False)
    args = parser.parse_args()

    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'cifar10'
    epochs = 25
    batch_size = 100

    kw = {
        'eigen_tol': args.tol,
        'sub_prob_max_iters': 50,
        'sub_prob_method': 'explicit'
    }

    path = os.path.dirname(__file__)
    file_name = f'explicit_{str(args.tol).replace(".","")}'

    output, model = imc(dataset=dataset, optim_method='sarc', epochs=epochs,
                            batch_size=batch_size, return_model=True, validate=1,
                            sample_rate=0.03, **kw)

    data_dir = os.path.join(path, f'data/{file_name}.json')

    with open(data_dir, 'w') as file:
        json.dump(output, file)

    if args.save_model:
        data_dir = os.path.join(path, f'models/{file_name}.pt')

        torch.save(model.state_dict(), data_dir)

    print('Test finished.')
