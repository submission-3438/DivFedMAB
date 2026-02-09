import numpy as np
import copy

class Client(object):
    
    def __init__(self, id, group=None, train_data={'x':[],'y':[]}, eval_data={'x':[],'y':[]}, model=None):
        self.model = model
        self.id = id # integer
        self.group = group
        self.train_data = {k: np.array(v) for k, v in train_data.items()}
        self.eval_data = {k: np.array(v) for k, v in eval_data.items()}


        # --------------------------------------------------
        # CIFAR-10 NORMALIZATION (SAFE & AUTOMATIC)
        # --------------------------------------------------
        def normalize_cifar(X):
            X = X.astype(np.float32)

            # scale to [0,1] if needed
            if X.max() > 1.0:
                X /= 255.0

            # CIFAR-10 channel-wise normalization
            mean = np.array([0.4914, 0.4822, 0.4465])
            std  = np.array([0.2023, 0.1994, 0.2010])

            return (X - mean) / std

        # Normalize training data if CIFAR-10
        if self.train_data['x'].ndim == 4 and self.train_data['x'].shape[-1] == 3:
            self.train_data['x'] = normalize_cifar(self.train_data['x'])

        # Normalize eval data if CIFAR-10
        if self.eval_data['x'].ndim == 4 and self.eval_data['x'].shape[-1] == 3:
            self.eval_data['x'] = normalize_cifar(self.eval_data['x'])

        # --------------------------------------------------
        # CIFAR-10 NORMALIZATION (SAFE & AUTOMATIC)
        # --------------------------------------------------



        self.num_samples = len(self.train_data['y'])
        self.test_samples = len(self.eval_data['y'])
        self.updatevec = np.append(model.get_params()[0].flatten(), model.get_params()[1])

    def set_params(self, model_params):
        '''set model parameters'''
        self.model.set_params(model_params)

    def get_params(self):
        '''get model parameters'''
        return self.model.get_params()

    def get_grads(self, model_len):
        '''get model gradient'''
        return self.model.get_gradients(self.train_data, model_len)

    def solve_grad(self):
        '''get model gradient with cost'''
        bytes_w = self.model.size
        grads = self.model.get_gradients(self.train_data)
        comp = self.model.flops * self.num_samples
        bytes_r = self.model.size
        return ((self.num_samples, grads), (bytes_w, comp, bytes_r))

    def solve_inner(self, num_epochs=1, batch_size=10):
        '''Solves local optimization problem
        
        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp, grads = self.model.solve_inner(self.train_data, num_epochs, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r), grads

    def solve_iters(self, num_iters=1, batch_size=10):
        '''Solves local optimization problem

        Return:
            1: num_samples: number of samples used in training
            1: soln: local optimization solution
            2: bytes read: number of bytes received
            2: comp: number of FLOPs executed in training process
            2: bytes_write: number of bytes transmitted
        '''

        bytes_w = self.model.size
        soln, comp = self.model.solve_iters(self.train_data, num_iters, batch_size)
        bytes_r = self.model.size
        return (self.num_samples, soln), (bytes_w, comp, bytes_r)

    def train_error_and_loss(self):
        tot_correct, loss = self.model.test(self.train_data)
        return tot_correct, loss, self.num_samples


    def test(self):
        '''tests current model on local eval_data

        Return:
            tot_correct: total #correct predictions
            test_samples: int
        '''
        tot_correct, loss = self.model.test(self.eval_data)
        return tot_correct, self.test_samples
