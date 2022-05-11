from core.problems import imageClassification as imc
import json
import os

if __name__=='__main__':
    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'mnist'
    epochs = 10
    batch_size = 100

    kw = {'lr': 0.01, 'momentum': 0.9}

    output = imc(dataset=dataset, optim_method='sgd', epochs=epochs,
                    batch_size=batch_size, return_model=False, validate=1, **kw)

    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, 'data/sgd.json')

    with open(data_dir, 'w') as file:
        json.dump(output, file)

    print('Test finished.')
