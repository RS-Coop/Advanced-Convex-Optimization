from core.problems import imageClassification as imc
import json
import os

if __name__=='__main__':
    print('Beginning test...')

    #Setup hyper-parameters
    dataset = 'mnist'
    epochs = 10
    batch_size = 100

    kw = {
        'eigen_tol': 1e-2,
        'sub_prob_max_iters': 50,
        'sub_prob_method': 'explicit'
    }

    output = imc(dataset=dataset, optim_method='sarc', epochs=epochs,
                    batch_size=batch_size, return_model=False, validate=1,
                    sample_rate=0.03, **kw)

    path = os.path.dirname(__file__)
    data_dir = os.path.join(path, 'data/explicit.json')

    with open(data_dir, 'w') as file:
        json.dump(output, file)

    print('Test finished.')
