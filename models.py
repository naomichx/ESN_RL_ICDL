import scipy.sparse as sp
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, Input, RLS
from reservoirpy.mat_gen import Initializer, uniform, random_sparse, normal, bernoulli
rpy.verbosity(0)
import numpy as np
import json
import os
import random
from utils import build, build_separate_input, build_forward
random.seed(1)
np.random.seed(1)



class M_O:
    def __init__(self, seed, filename='M_0.json', n_position=4, hyperparam_optim=False,  lr=None, sr=None,
                 rc_connectivity=None,  input_connectivity=None,
                  eta=None, beta=None, fb_connectivity=None,output_connectivity=None,
                 decay=None,separate_input=False):
        """
        This class implements the Echo State Network Model trained
        with online RL.

        parameters:

                units: int
                        number of reservoir neurons
                sr: float
                        spectral radius
                lr: float
                        leak rate
                fb_scaling: float
                        feedback scaling
                input_scaling: float
                        input scaling
                noise_rc: float
                        reservoir noise
                rc_connectivity: float
                        reservoir connectivity
                input_connectivity: float
                        input connectivity
                fb_connectivity: float
                        feedback connectivity

                beta: int
                      inverse temperature
                eta: float
                    learning rate of the RL model
                r_th: float

        """

        self.filename = filename
        self.seed = seed

        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity, eta=eta, beta=beta,
                   fb_connectivity=fb_connectivity, output_connectivity=output_connectivity, decay=decay)
        self.all_p = None
        self.esn_output = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.record_reservoir_states = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = False
        self.n_res = 1
        self.separate_input = separate_input


    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None,
              input_connectivity=None, output_connectivity=None, eta=None, beta=None,
              fb_connectivity=None, decay=None):

        _ = self.parameters
        self.units = _['ESN']['n_units']

        if hyperparam_optim:

            self.r_th = _['RL']['r_th']
            self.reward = _['RL']['reward']
            self.penalty = _['RL']['penalty']
            self.noise_rc = _['ESN']['noise_rc']
            self.fb_scaling = _['ESN']['fb_scaling']
            self.fb_connectivity = fb_connectivity[0]
            self.input_scaling = _['ESN']['input_scaling']
            self.output_connectivity = output_connectivity
            self.lr = lr[0]
            self.sr = sr[0]
            self.rc_connectivity = rc_connectivity[0]
            self.input_connectivity = input_connectivity[0]
            self.eta = eta
            self.beta = beta
            self.decay = decay
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.r_th = _['RL']['r_th']
            self.reward = _['RL']['reward']
            self.penalty = _['RL']['penalty']
            self.decay = _['RL']['decay']
            #self.units = _['ESN']['n_units']
            self.lr =_['ESN']['lr']
            self.sr = _['ESN']['sr']
            self.rc_connectivity = _['ESN']['rc_connectivity']
            self.noise_rc = _['ESN']['noise_rc']
            self.fb_scaling = _['ESN']['fb_scaling']
            self.input_connectivity = _['ESN']['input_connectivity']
            self.fb_connectivity = _['ESN']['fb_connectivity']
            self.input_scaling = _['ESN']['input_scaling']
            self.output_connectivity = _['ESN']['output_connectivity']

        self.W = normal(loc=0,
                             scale=self.sr ** 2 / (self.rc_connectivity * self.units),
                             seed=self.seed)

        print('n_units', self.units)
        self.reservoir = Reservoir(units=self.units, lr=self.lr, sr=self.sr,
                                   input_scaling=self.input_scaling,
                                   W=self.W,
                                   rc_connectivity=self.rc_connectivity, noise_rc=self.noise_rc,
                                   fb_scaling=self.fb_scaling,
                                   input_connectivity=self.input_connectivity,
                                   fb_connectivity=self.fb_connectivity, seed=self.seed,
                                   activation= 'tanh')
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir <<= self.readout
        self.esn = self.reservoir >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)


    def softmax(self, x):
        """
            Return the softmax of x corresponding to probabilities, the sum of all
            probabilities is equal to 1.

            parameters:
                x: array of shape (n_output,)

                    """
        all_p = np.exp(self.beta * x)
        index_inf = None
        for i in range(len(all_p)):
            if np.isinf(all_p[i]):
                index_inf = i
        if index_inf is not None:
            all_p = [0 for i in range(len(all_p))]
            all_p[index_inf] = 1
        elif all(k == 0 for k in list(all_p)):
            index_max = np.argmax(x)
            all_p = [0 for i in range(len(all_p))]
            all_p[index_max] = 1
        else:
            all_p = [all_p[i] / np.sum(np.exp(self.beta * x), axis=0) for i in range(len(all_p))]
        return all_p


    def select_choice(self):
        """
            Compute the choice of the ESN model
            """
        p = np.random.random()
        if p < self.epsilon:
            self.choice = np.random.choice(4)
        else:
            self.choice = np.argmax(self.esn_output)
        self.epsilon *= self.decay


    def process(self, trial_chronogram, count_record=None, record_output= False, record_reservoir_states=False):
        self.record_reservoir_states[count_record] = {}
        for i in range(self.n_res):
            self.record_reservoir_states[count_record]['reservoir_{}'.format(str(i))] = []
        for i in range(len(trial_chronogram)):
            self.esn_output = self.esn.call(trial_chronogram[i].ravel())[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
            if record_reservoir_states:
                if self.n_res == 1:
                    self.record_reservoir_states[count_record]['reservoir_0'].append(self.reservoir.state()[0])
                else:
                    for j in range(self.n_res):
                        self.record_reservoir_states[count_record]['reservoir_{}'.format(str(j))].append(self.reservoir[j].state()[0])
        self.select_choice()


    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = np.array(self.readout.params['Wout'].todense())
        else:
            W_out = np.array(self.readout.params['Wout'])
        if self.flag:
            self.mask = W_out != 0
            self.flag = False
        r = self.reservoir.state()

        W_out[:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice])* (r[0][:] - self.r_th))
        W_out = W_out*self.mask
        self.all_W_out.append(W_out)

        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class M_1(M_O):
    def __init__(self, seed, filename='wide_model.json', n_position=4, hyperparam_optim=False, lr=None, sr=None,
                  rc_connectivity=None, input_connectivity=None, eta=None, beta=None, decay=None, i_sim=None,
                 fb_connectivity=None, output_connectivity=None, n_units=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.setup(hyperparam_optim=hyperparam_optim,  lr=lr, sr=sr,
                   rc_connectivity=rc_connectivity,
                   input_connectivity=input_connectivity,
                   eta=eta, beta=beta, decay=decay, i_sim=i_sim, fb_connectivity=fb_connectivity,
                   output_connectivity=output_connectivity,
                   units=n_units)

        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.record_reservoir_states = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = True

        print('Separate input:', self.separate_input)

    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, i_sim=None, fb_connectivity=None, output_connectivity=None, units=None):

        _ = self.parameters

        self.i_sim = i_sim
        self.r_th = _['RL']['r_th']
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']

        self.output_connectivity = _['Readout']['output_connectivity']
        self.units = [250, 250]
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.fb_connectivity = []
        self.W = []

        if hyperparam_optim:
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(2):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])

                self.fb_connectivity.append(fb_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units[i]),
                                     seed=self.seed))
        else:
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(2):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i + 1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units[i]),
                                     seed=self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), 2)
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir = {}
        for i in range(2):
            self.reservoir[i] = Reservoir(units=self.units[i], lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
                                          activation='tanh',
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))
            self.reservoir[i] <<= self.readout
        self.esn = [self.reservoir[0], self.reservoir[1]] >> self.readout
        np.random.seed(self.seed)
        random.seed(self.seed)

    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False,
                record_reservoir_states=False):
        if record_reservoir_states:
            self.record_reservoir_states[count_record] = {}
            for i in range(2):
                self.record_reservoir_states[count_record]['reservoir_{}'.format(str(i))] = []
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_1': trial_chronogram_late[i].ravel()})[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
            if record_reservoir_states:
                for k in range(2):

                    self.record_reservoir_states[count_record]['reservoir_{}'.format(str(k))].append(self.reservoir[k].state()[0])
        self.select_choice()

    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']
        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False
        r_states = {}
        for i in range(2):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        W_out_dict[0] = np.array(W_out[0: self.units[0]])
        W_out_dict[1] = np.array(W_out[self.units[0]:self.units[0]+ self.units[1]])

        for i in range(2):
            W_out_dict[i][:, choice] += np.array(
                self.eta * (reward - self.softmax(self.esn_output)[choice]) * (r_states[i][0][:] - self.r_th))
        W_out = np.concatenate((W_out_dict[0], W_out_dict[1]))
        W_out = W_out * self.mask
        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class M_2(M_O):
    def __init__(self, seed, filename, n_position=4, i_sim=None,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None,fb_connectivity=None, output_connectivity=None,fb_scaling=None,
                 r_th=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.n_res = 4
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, i_sim=i_sim,
                   rc_connectivity=rc_connectivity, input_connectivity=input_connectivity,
                   eta=eta, beta=beta, decay=decay, fb_connectivity=fb_connectivity,
                   output_connectivity=output_connectivity,fb_scaling=fb_scaling, r_th=r_th)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.record_reservoir_states = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = True


    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, fb_connectivity=None, output_connectivity=None,i_sim=None, r_th = None,
              fb_scaling=None):

        self.i_sim = i_sim
        _ = self.parameters

        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']
        self.fb_scaling = _['ESN_1']['fb_scaling']

        self.output_connectivity = _['Readout']['output_connectivity']
        self.noise_rc = _['ESN_1']['noise_rc']

        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']

        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.fb_connectivity = []
        self.W = []

        self.units = [125, 125, 125,125]
        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), self.n_res)

        if hyperparam_optim:
            self.r_th = r_th
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(self.n_res):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units[i]),
                                     seed=self.seed))
        else:
            self.r_th = _['RL']['r_th']
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(self.n_res):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i + 1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units[i]),
                                     seed=self.seed))
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir = {}
        for i in range(4):
            self.reservoir[i] = Reservoir(units=self.units[i], lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
                                          activation='tanh',
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))
            self.reservoir[i] <<= self.readout
        pathway_1 = self.reservoir[0] >> self.reservoir[1]
        pathway_2 = self.reservoir[2] >> self.reservoir[3]

        self.esn = [self.reservoir[0], self.reservoir[2], pathway_1, pathway_2] >> self.readout
        #np.random.seed(self.seed)
        #random.seed(self.seed)

    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False,
                record_reservoir_states=False):
        if record_reservoir_states:
            self.record_reservoir_states[count_record] = {}
            for i in range(self.n_res):
                self.record_reservoir_states[count_record]['reservoir_{}'.format(str(i))] = []
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_2': trial_chronogram_late[i].ravel()})[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
            if record_reservoir_states:
                for k in range(self.n_res):
                    self.record_reservoir_states[count_record]['reservoir_{}'.format(str(k))].append(self.reservoir[k].state()[0])
        self.select_choice()


    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(self.n_res):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        begin = 0
        end = 0
        for i in range(self.n_res):
            if i > 0:
                begin += self.units[i-1]
            end += self.units[i]
            #W_out_dict[i] = np.array(W_out[i*self.units:(i+1)*self.units])
            W_out_dict[i] = np.array(W_out[begin:end])
            W_out_dict[i][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]))*(r_states[i][0][:] - self.r_th)

        W_out = np.concatenate((W_out_dict[0], W_out_dict[1], W_out_dict[2], W_out_dict[3]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class M_3(M_O):
    def __init__(self, seed, filename, n_position=4, i_sim=None,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 eta=None, beta=None, decay=None,fb_connectivity=None, output_connectivity=None,fb_scaling=None,
                 r_th=None):
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.n_res = 6
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr, i_sim=i_sim,
                   rc_connectivity=rc_connectivity, input_connectivity=input_connectivity,
                   eta=eta, beta=beta, decay=decay, fb_connectivity=fb_connectivity,
                   output_connectivity=output_connectivity,fb_scaling=fb_scaling, r_th=r_th)
        self.all_p = None
        self.choice = None
        self.all_W_out = []
        self.record_output_activity = {}
        self.record_reservoir_states = {}
        self.flag = True
        self.mask = None
        self.epsilon = 1
        self.separate_input = True


    def setup(self, hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
              eta=None, beta=None, decay=None, fb_connectivity=None, output_connectivity=None,i_sim=None, r_th = None,
              fb_scaling=None):
        self.i_sim = i_sim
        _ = self.parameters
        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.activation_func = _['ESN_1']['activation']
        self.fb_scaling = _['ESN_1']['fb_scaling']
        self.output_connectivity = _['Readout']['output_connectivity']
        self.noise_rc = _['ESN_1']['noise_rc']
        self.fb_connectivity = _['ESN_1']['fb_connectivity']
        self.input_scaling = _['ESN_1']['input_scaling']
        self.lr = []
        self.sr = []
        self.rc_connectivity = []
        self.input_connectivity = []
        self.fb_connectivity = []
        self.W = []
        self.units = [83, 83, 84,83,83,84]
        random.seed(self.seed)
        np.random.seed(self.seed)
        seeds = random.sample(range(1, 100), self.n_res)

        if hyperparam_optim:
            self.r_th = r_th
            self.eta = eta
            self.beta = beta
            self.decay = decay
            self.output_connectivity = output_connectivity
            for i in range(self.n_res):
                self.lr.append(lr[i])
                self.sr.append(sr[i])
                self.rc_connectivity.append(rc_connectivity[i])
                self.fb_connectivity.append(fb_connectivity[i])
                self.input_connectivity.append(input_connectivity[i])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units[i]),
                                     seed=self.seed))
        else:
            self.r_th = _['RL']['r_th']
            self.eta = _['RL']['eta']
            self.beta = _['RL']['beta']
            self.decay = _['RL']['decay']
            self.output_connectivity = _['Readout']['output_connectivity']
            for i in range(self.n_res):
                self.lr.append(_['ESN_{}'.format(str(i + 1))]['lr'])
                self.sr.append(_['ESN_{}'.format(str(i + 1))]['sr'])
                self.rc_connectivity.append(_['ESN_{}'.format(str(i + 1))]['rc_connectivity'])
                self.fb_connectivity.append(_['ESN_{}'.format(str(i + 1))]['fb_connectivity'])
                self.input_connectivity.append(_['ESN_{}'.format(str(i + 1))]['input_connectivity'])
                self.W.append(normal(loc=0,
                                     scale=self.sr[-1] ** 2 / (self.rc_connectivity[-1] * self.units[i]),
                                     seed=self.seed))
        self.readout = Ridge(self.n_position,
                             Wout=uniform(low=0, high=1, connectivity=self.output_connectivity, seed=self.seed))
        self.reservoir = {}
        for i in range(self.n_res):
            self.reservoir[i] = Reservoir(units=self.units[i], lr=self.lr[i], sr=self.sr[i],
                                          input_scaling=self.input_scaling, W=self.W[i],
                                          rc_connectivity=self.rc_connectivity[i],
                                          noise_rc=self.noise_rc,
                                          fb_scaling=self.fb_scaling,
                                          input_connectivity=self.input_connectivity[i],
                                          fb_connectivity=self.fb_connectivity[i], seed=seeds[i],
                                          activation='tanh',
                                          name='reservoir_{}'.format(self.i_sim) + '_{}'.format(str(i)))
            self.reservoir[i] <<= self.readout
        pathway_1 = self.reservoir[0] >> self.reservoir[1] >> self.reservoir[2]
        pathway_2 = self.reservoir[3] >> self.reservoir[4] >> self.reservoir[5]

        self.esn = [self.reservoir[0], self.reservoir[1], pathway_1, self.reservoir[3], self.reservoir[4],
                    pathway_2] >> self.readout
        #np.random.seed(self.seed)
        #random.seed(self.seed)

    def process(self, trial_chronogram_early, trial_chronogram_late, count_record=None, record_output=False,
                record_reservoir_states=False):
        if record_reservoir_states:
            self.record_reservoir_states[count_record] = {}
            for i in range(self.n_res):
                self.record_reservoir_states[count_record]['reservoir_{}'.format(str(i))] = []
        for i in range(len(trial_chronogram_early)):
            self.esn_output = self.esn.call({'reservoir_{}'.format(self.i_sim) + '_0': trial_chronogram_early[i].ravel(),
                                            'reservoir_{}'.format(self.i_sim) + '_3': trial_chronogram_late[i].ravel()})[0]
            if record_output:
                self.record_output_activity[count_record]['output_activity'].append(self.readout.state()[0])
            if record_reservoir_states:
                for k in range(self.n_res):
                    self.record_reservoir_states[count_record]['reservoir_{}'.format(str(k))].append(self.reservoir[k].state()[0])
        self.select_choice()


    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']

        if self.flag:
            self.mask = np.array(W_out != 0)
            self.flag = False

        r_states = {}
        for i in range(self.n_res):
            r_states[i] = self.reservoir[i].state()

        W_out_dict = {}
        begin = 0
        end = 0
        for i in range(self.n_res):
            if i > 0:
                begin += self.units[i-1]
            end += self.units[i]
            #W_out_dict[i] = np.array(W_out[i*self.units:(i+1)*self.units])
            W_out_dict[i] = np.array(W_out[begin:end])

            W_out_dict[i][:, choice] += np.array(self.eta * (reward - self.softmax(self.esn_output)[choice]))*(r_states[i][0][:] - self.r_th)

        W_out = np.concatenate((W_out_dict[0], W_out_dict[1], W_out_dict[2], W_out_dict[3],W_out_dict[4],W_out_dict[5]))
        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        for i in range(self.n_position):
            col_norm = np.linalg.norm(W_out[:, i])
            if col_norm != 0:
                W_out[:, i] = W_out[:, i]/col_norm
        self.readout.params['Wout'] = sp.csr_matrix(W_out)


class M_plus(M_O):
    def __init__(self, seed, filename, separate_input, n_position=4,
                 hyperparam_optim=False, lr=None,
                 sr=None, connect_limit=None, connect_prob=None, angle=None,
                 eta=None, decay=None,
                 fb_connectivity=None, output_connectivity=None, r_th=None, beta=None):
        """
        This class implements the Echo State Network Model trained
        with online RL.

        parameters:

                units: int
                        number of reservoir neurons
                sr: float
                        spectral radius
                lr: float
                        leak rate
                fb_scaling: float
                        feedback scaling
                input_scaling: float
                        input scaling
                noise_rc: float
                        reservoir noise
                connect_limit: float
                        reservoir connectivity
                connect_prob: float
                        input connectivity
                fb_connectivity: float
                        feedback connectivity

                beta: int
                      inverse temperature
                eta: float
                    learning rate of the RL model
                r_th: float

        """
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.epsilon = 1
        self.separate_input = separate_input
        print('Separate input :', separate_input)
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr,
                   connect_limit=connect_limit, beta=beta,
                   connect_prob=connect_prob, angle=angle, eta=eta,
                   decay=decay,
                   fb_connectivity=fb_connectivity, output_connectivity=output_connectivity, r_th=r_th)
        self.all_p = None
        self.esn_output = None
        self.choice = None
        self.softmax_issue = False
        self.all_W_out = []
        self.record_output_activity = [[] for i in range(250)]
        self.record_output_activity = {}
        self.record_reservoir_states = {}
        self.n_res = 1

    def setup(self, hyperparam_optim=False, lr=None,
              sr=None, connect_limit=None, connect_prob=None, angle=None,
              eta=None, decay=None, epsilon=None,
              fb_connectivity=None, output_connectivity=None, r_th=None, beta=None):

        _ = self.parameters

        if not hyperparam_optim:
            self.eta = _['RL']['eta']
            self.decay = _['RL']['decay']
            self.lr = _['ESN']['lr']
            self.sr = _['ESN']['sr']
            self.connect_limit = _['ESN']['connect_limit']
            self.connect_prob = _['ESN']['connect_prob']
            self.angle = _['ESN']['angle']
            self.fb_connectivity = _['ESN']['fb_connectivity']
            self.output_connectivity = _['ESN']['output_connectivity']
            self.r_th = _['RL']['r_th']
            self.beta = _['RL']['beta']
        else:
            self.eta = eta
            self.decay = decay
            self.lr = lr[0]
            self.sr = sr
            self.connect_limit = connect_limit[0]
            self.connect_prob = connect_prob[0]
            self.angle = angle[0]
            self.fb_connectivity = fb_connectivity
            self.output_connectivity = output_connectivity
            self.r_th = r_th
            self.beta = beta

        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.units = _['ESN']['n_units']
        self.noise_rc = _['ESN']['noise_rc']
        self.fb_scaling = _['ESN']['fb_scaling']
        self.input_scaling = _['ESN']['input_scaling']

        np.random.seed(self.seed)
        random.seed(self.seed)

        cells_pos, W, W_in, W_out = build_forward(limit=self.connect_limit,
                                                  connect_prob=self.connect_prob,
                                                  angle=self.angle,
                                                  weight_scale=self.sr,
                                                  output_param=self.output_connectivity,
                                                  seed=self.seed)

        bias_arr = 0.1 * np.random.binomial(n=1, p=0.125, size=(self.units, 1))

        self.reservoir = Reservoir(units=self.units, lr=self.lr,
                                   input_scaling=self.input_scaling,
                                   W=W, Win=W_in, bias=bias_arr,
                                   noise_rc=self.noise_rc,
                                   fb_connectivity=self.fb_connectivity, fb_scaling=self.fb_scaling,
                                   activation='tanh',
                                   seed=self.seed)

        self.readout = Ridge(self.n_position)
        self.mask = (W_out != 0)
        self.reservoir <<= self.readout
        self.esn = self.reservoir >> self.readout

    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']
        r = self.reservoir.state()
        W_out[:, choice] = W_out[:, choice] + self.eta * (reward - self.softmax(self.esn_output)[choice]) * \
                           (r[0][:] - self.r_th)

        W_out = W_out * self.mask

        self.all_W_out.append(W_out)
        self.readout.params['Wout'] = W_out


class M_star(M_O):
    def __init__(self, seed, filename,n_position=4,
                 hyperparam_optim=False, lr=None,
                 sr=None, connect_limit=None, connect_prob=None, angle=None,
                 eta=None, decay=None,
                 fb_connectivity=None, output_connectivity=None, r_th=None, beta=None):
        """
        This class implements the Echo State Network Model trained
        with online RL.

        parameters:

                units: int
                        number of reservoir neurons
                sr: float
                        spectral radius
                lr: float
                        leak rate
                fb_scaling: float
                        feedback scaling
                input_scaling: float
                        input scaling
                noise_rc: float
                        reservoir noise
                connect_limit: float
                        reservoir connectivity
                connect_prob: float
                        input connectivity
                fb_connectivity: float
                        feedback connectivity

                beta: int
                      inverse temperature
                eta: float
                    learning rate of the RL model
                r_th: float

        """
        self.filename = filename
        self.seed = seed
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.n_position = n_position
        self.epsilon = 1
        self.setup(hyperparam_optim=hyperparam_optim, lr=lr, sr=sr,
                   connect_limit=connect_limit, beta=beta,
                   connect_prob=connect_prob, angle=angle, eta=eta,
                   decay=decay,
                   fb_connectivity=fb_connectivity, output_connectivity=output_connectivity, r_th=r_th)
        self.all_p = None
        self.esn_output = None
        self.choice = None
        self.softmax_issue = False
        self.all_W_out = []
        self.record_output_activity = [[] for i in range(250)]
        self.record_output_activity = {}
        self.record_reservoir_states = {}
        self.n_res = 1


    def setup(self, hyperparam_optim=False, lr=None,
              sr=None, connect_limit=None, connect_prob=None, angle=None,
              eta=None, decay=None, epsilon=None,
              fb_connectivity=None, output_connectivity=None, r_th=None,  beta=None):

        _ = self.parameters

        if not hyperparam_optim:
            self.eta = _['RL']['eta']
            self.decay = _['RL']['decay']
            self.lr = _['ESN']['lr']
            self.sr = _['ESN']['sr']
            self.connect_limit = _['ESN']['connect_limit']
            self.connect_prob = _['ESN']['connect_prob']
            self.angle = _['ESN']['angle']
            self.fb_connectivity = _['ESN']['fb_connectivity']
            self.output_connectivity = _['ESN']['output_connectivity']
            self.r_th = _['RL']['r_th']
            self.beta = _['RL']['beta']
        else:
            self.eta = eta
            self.decay = decay
            self.lr = lr
            self.sr = sr
            self.connect_limit = connect_limit
            self.connect_prob = connect_prob
            self.angle = angle
            self.fb_connectivity = fb_connectivity
            self.output_connectivity = output_connectivity
            self.r_th = r_th
            self.beta = beta

        self.reward = _['RL']['reward']
        self.penalty = _['RL']['penalty']
        self.units = _['ESN']['n_units']
        self.noise_rc = _['ESN']['noise_rc']
        self.fb_scaling = _['ESN']['fb_scaling']
        self.input_scaling = _['ESN']['input_scaling']

        np.random.seed(self.seed)
        random.seed(self.seed)

        cells_pos, W, W_in, W_out, domain = build_separate_input(limit=self.connect_limit,
                                                  connect_prob=self.connect_prob,
                                                  angle=self.angle,
                                                  weight_scale=self.sr,
                                                  output_param=self.output_connectivity,
                                                  seed=self.seed)

        bias_arr = 0.1 * np.random.binomial(n=1, p=0.125, size=(self.units, 1))
        lr_arr = [self.lr[int(domain[i])] for i in range(len(domain))]
        lr_arr = np.array(lr_arr)

        self.reservoir = Reservoir(units=self.units, lr=lr_arr,
                                   input_scaling=self.input_scaling,
                                   W=W, Win=W_in, bias=bias_arr,
                                   noise_rc=self.noise_rc,
                                   fb_connectivity=self.fb_connectivity, fb_scaling=self.fb_scaling,
                                   activation='tanh',
                                   seed=self.seed)

        self.readout = Ridge(self.n_position)
        self.mask = (W_out != 0)
        self.reservoir <<= self.readout
        self.esn = self.reservoir >> self.readout

    def train(self, reward, choice):
        """
            Train the readout of the ESN model.
            parameters:
                reward: float
                        reward obtained after the model choice.
            """
        if sp.issparse(self.readout.params['Wout']):
            W_out = self.readout.params['Wout'].todense()
        else:
            W_out = self.readout.params['Wout']
        r = self.reservoir.state()

        #W_out[:, choice] = W_out[:, choice] + self.eta * (reward - self.esn_output[choice]) * \
         #                  (r[0][:] - self.r_th)

        W_out[:, choice] = W_out[:, choice] + self.eta * (reward - self.softmax(self.esn_output)[choice]) * \
                           (r[0][:] - self.r_th)

        W_out = W_out * self.mask

        self.all_W_out.append(W_out)

        # Remove the normalization for the moment.
        # for i in range(self.n_position):
        #    col_norm = np.linalg.norm(W_out[:, i])
        #    if col_norm != 0:
        #        W_out[:, i] = W_out[:, i]/col_norm

        self.readout.params['Wout'] = W_out
        # self.readout.params['Wout'] = sp.csr_matrix(updated_W_out)

