from task import Task
from models import *


class Experiment:
    """
    This class run the whole experiment.

    """
    def __init__(self, seed, model_file, task_file, model_type,
                 hyperparam_optim=False, lr=None, sr=None, rc_connectivity=None, input_connectivity=None,
                 fb_connectivity=None, eta=None, beta=None,  output_connectivity=None, r_th=None,
                decay=None, connect_limit=None, connect_prob=None, angle=None, testing=None, i_sim=0):

        random.seed(seed)
        np.random.seed(seed)
        self.model_file = model_file
        self.task_file = task_file
        self.task = Task(filename=task_file)
        self.n_trial = len(self.task.trials)
        self.model_type = model_type
        self.n_input_time_1 = self.task.n_input_time_1
        self.n_input_time_2 = self.task.n_input_time_2
        self.n_sessions = self.task.n_session
        self.nb_train = self.task.nb_train
        self.seeds = random.sample(range(1, 100), self.n_sessions)

        self.trial_counter = 0
        self.success_counter = 0
        self.count_record = 0

        self.hyperparam_optim = hyperparam_optim
        self.lr = lr
        self.sr = sr
        self.connect_limit = connect_limit
        self.angle = angle
        self.connect_prob = connect_prob
        self.rc_connectivity = rc_connectivity
        self.output_connectivity = output_connectivity
        self.input_connectivity = input_connectivity
        self.fb_connectivity = fb_connectivity
        self.eta = eta
        self.beta = beta
        self.decay = decay
        self.r_th = r_th
        self.i_sim = i_sim

        self.testing = testing
        self.model_type = model_type
        self.init_model(seed)
        self.initialize_storage()

    def initialize_storage(self):
        """
           Initialize the storage arrays and dictionaries for
           during the experiment execution.
        """

        # Create array to store data during training phase
        self.success_array = []
        self.success_array_best_first = []
        self.success_array_best_last = []
        self.legal_choices_array = []
        self.all_trials = {
            'trials': [],
            'delay': [],
            'input_time_1': [],
            'input_time_2': []
        }
        # Create array to store data during testing phase
        if self.testing:
            self.success_array_testing = []
            self.success_array_best_first_testing = []
            self.success_array_best_last_testing = []
            self.legal_choices_array_testing = []
            self.all_trials_testing = {}
            self.all_trials_testing = {
                'trials': [],
                'delay': [],
                'input_time_1': [],
                'input_time_2': [],
                'best_first': []
            }

    def store_training_results(self, task, trial, model, choice):
        """
            Store the results of the final choice of the model: 1 if it made the right choice, 0 otherwise.

            Parameters:
                task: Task object
                model: Model object
                trial: numpy array of shape (n_cue, n_positions)
                        Current trial of the task.
                choice: int
                        The choice made by the model.
                test_count: bool, optional
                            If True, store the results in the testing arrays.
            """

        # Store trial information
        self.all_trials['trials'].append(trial)
        self.all_trials['delay'].append(task.n_input_delay)
        self.all_trials['input_time_1'].append(task.n_input_time_1)
        self.all_trials['input_time_2'].append(task.n_input_time_2)

        # Check if the choice is legal and store the result
        is_legal = task.is_legal_choice(trial, choice=choice)
        self.legal_choices_array.append(int(is_legal))

        # Check if the choice is correct and store the result
        is_correct = model.choice == task.get_best_choice(trial, model.reward)
        self.success_array.append(int(is_correct))

        # Store the result in the appropriate best choice array
        if is_correct:
            if task.best_trial_first:
                self.success_array_best_first.append(1)
            else:
                self.success_array_best_last.append(1)
        else:
            if task.best_trial_first:
                self.success_array_best_first.append(0)
            else:
                self.success_array_best_last.append(0)

        # Store the best cue information
        self.model.record_output_activity[self.count_record]['trial_info']['best_cue_first'] = task.best_trial_first

        # Store the rewards for all choices
        for c in [0, 1, 2, 3]:
            reward = task.get_reward(trial, c, model.parameters['RL']['reward'], model.parameters['RL']['penalty'])
            self.model.record_output_activity[self.count_record]['trial_info'][c] = reward

    def store_testing_results(self, task, trial, model, choice):
        """
        Store the results of the final choice of the model during testing: 1 if it made the right choice, 0 otherwise.

        Parameters:
            task: Task object
            model: Model object
            trial: numpy array of shape (n_cue, n_positions)
                    Current trial of the task.
            choice: int
                    The choice made by the model.
        """
        # Store trial information
        self.all_trials_testing['trials'].append(trial)
        self.all_trials_testing['delay'].append(task.n_input_delay)
        self.all_trials_testing['input_time_1'].append(task.n_input_time_1)
        self.all_trials_testing['input_time_2'].append(task.n_input_time_2)

        # Check if the choice is legal and store the result
        is_legal = task.is_legal_choice(trial, choice=choice)
        self.legal_choices_array_testing.append(int(is_legal))

        # Check if the choice is correct and store the result
        is_correct = model.choice == task.get_best_choice(trial, model.reward)
        self.success_array_testing.append(int(is_correct))

        # Store the result in the appropriate best choice array
        if is_correct:
            if task.best_trial_first:
                self.success_array_best_first_testing.append(1)
            else:
                self.success_array_best_last_testing.append(1)
        else:
            if task.best_trial_first:
                self.success_array_best_first_testing.append(0)
            else:
                self.success_array_best_last_testing.append(0)

        # Store the best cue information
        self.model.record_output_activity[self.count_record]['trial_info']['best_cue_first'] = task.best_trial_first

        # Store the rewards for all choices
        for c in [0, 1, 2, 3]:
            reward = task.get_reward(trial, c, model.parameters['RL']['reward'], model.parameters['RL']['penalty'])
            self.model.record_output_activity[self.count_record]['trial_info'][c] = reward

    def init_model(self, seed):
        if self.model_type == "M_0":
            print('Initialise M_0 model..')
            self.model = M_O(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                 rc_connectivity=self.rc_connectivity, input_connectivity=self.input_connectivity,
                                 eta=self.eta, beta=self.beta, decay=self.decay,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity,
                                 separate_input=True)
        elif self.model_type == "M_plus":
            print('Initialise M_plus model..')
            self.model = M_plus(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    connect_limit=self.connect_limit,
                                    connect_prob=self.connect_prob,
                                    angle = self.angle,
                                    eta=self.eta, beta=self.beta, decay=self.decay,
                                    output_connectivity=self.output_connectivity,
                                    fb_connectivity=self.fb_connectivity,
                                    r_th=self.r_th, separate_input = True)

        elif self.model_type == 'M_1':
            print('Initialise M_1 model..')
            self.model = M_1(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay, i_sim=self.i_sim,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity,
                                             )

        elif self.model_type == 'M_2':
            print('Initialise M_2 model..')
            self.model = M_2(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity, r_th=self.r_th,
                                      eta=self.eta, beta=self.beta, decay=self.decay, i_sim =self.i_sim,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity)

        elif self.model_type == "M_3":
            print('Initialise M_3 model..')
            self.model = M_3(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    rc_connectivity=self.rc_connectivity,
                                    input_connectivity=self.input_connectivity,
                                      eta=self.eta, beta=self.beta, decay=self.decay,i_sim =self.i_sim,
                                 output_connectivity=self.output_connectivity, fb_connectivity=self.fb_connectivity,
                                                       r_th=self.r_th)

        elif self.model_type == 'M_star':
            print('Initialise M* model..')
            self.model = M_star(filename=self.model_file, seed=seed, n_position=self.task.n_pos,
                                    hyperparam_optim=self.hyperparam_optim, lr=self.lr, sr=self.sr,
                                    connect_limit=self.connect_limit,
                                    connect_prob=self.connect_prob,
                                    angle = self.angle,
                                    eta=self.eta, beta=self.beta, decay=self.decay,
                                    output_connectivity=self.output_connectivity,
                                    fb_connectivity=self.fb_connectivity,
                                    r_th=self.r_th)

    def process_one_trial(self, trial, record_output=True,record_reservoir_states=False, force_order=None):
        self.count_record += 1
        self.model.record_output_activity[self.count_record] = {}
        self.model.record_output_activity[self.count_record]['output_activity'] = []
        self.model.record_output_activity[self.count_record]['trial_info'] = {}

        if self.model_type == 'M_0' or self.model_type == 'M_0_bis' or self.model_type == 'M_star':
            trial_with_chronogram = self.task.get_trial_with_chronogram(trial, separate_pathway=False)
            self.model.process(trial_with_chronogram, count_record=self.count_record,
                               record_output=record_output,
                               record_reservoir_states=record_reservoir_states)
        else:
            trial_with_chronogram_early, trial_with_chronogram_late = self.task.get_trial_with_chronogram(trial,
                                                                                                          separate_pathway=True)
            self.model.process(trial_with_chronogram_early, trial_with_chronogram_late, count_record=self.count_record,
                           record_output=record_output, record_reservoir_states=record_reservoir_states)

    def run(self):
        """
       Run the experiment n_sessions times. It is either a generalization test or a normal run.
       At each session, the trial list is first shuffled, before the model executes the task for each trial.
       parameters:
               - n_unseen: int
                 used when self.val_gen_test=True, corresponds to the number of unseen combinations that are not seen
                 by the model during training. Those unseen trials will be used only during the testing phase.
               """
        trials = self.task.trials
        trial_indexes = [i for i in range(trials.shape[0])]
        for i in range(self.nb_train):
            self.task.set_random_timings()
            if int(i % len(trials)) == 0:
                both = list(zip(trials, trial_indexes))
                random.shuffle(both)
                trials, trial_indexes = zip(*both)
                trials, trial_indexes = np.array(trials), np.array(trial_indexes)
            trial = trials[int(i % len(trials))]
            self.process_one_trial(trial, record_reservoir_states=False)
            reward = self.task.get_reward(trial, self.model.choice, self.model.reward, self.model.penalty)
            self.model.train(reward, self.model.choice)
            self.store_training_results(self.task, trial, self.model, self.model.choice)

        if self.testing:
            for i in range(1000):
                self.task.set_random_timings()
                if int(i % len(trials)) == 0:
                    both = list(zip(trials, trial_indexes))
                    random.shuffle(both)
                    trials, trial_indexes = zip(*both)
                    trials, trial_indexes = np.array(trials), np.array(trial_indexes)
                trial = trials[int(i % len(trials))]
                self.process_one_trial(trial, record_reservoir_states=True)
                self.store_testing_results(self.task, trial, self.model, self.model.choice)

