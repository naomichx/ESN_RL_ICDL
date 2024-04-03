import numpy as np
from math import comb, factorial
from itertools import permutations, combinations, product
import os
import json
import random
import pylab as p


def boards(p, q, n):
    comb = combinations(range(p), n)
    perm = permutations(range(q), n)
    coords = product(comb, perm)

    def make_board(c):
        arr = np.zeros((p, q), dtype=int)
        arr[c[0], c[1]] = 1
        return arr
    return map(make_board, coords)

def num_boards(p, q, n):
    return comb(p, n) * comb(q, n) * factorial(n)


class Task:
    def __init__(self, filename='task_test.json'):
        """
        This class implements a task where n stimuli (chosen among
        n_cue) are placed at n locations (chosen among n_pos). An
        agent
        has to pick a location and get the reward associated with
        the
        cue at that location. Each trial can hence be summarized
        by a
        (n_cue,n_pos) matrix with exactly two 1s inside (with no
        line nor column having more than one 1).

        Parameters:

          n_cue : int

            total number of cues

          n_pos : int

            total number of positions

          n_choice: int

            number of choices in a trial

          reward_probabilities : list

            Reward probability associated with each cue
        """
        self.filename = filename
        with open(os.path.join(os.path.dirname(__file__), filename)) as f:
            self.parameters = json.load(f)
        self.setup()
        self.unseen_trials = None
        assert len(self.reward_probabilities) == self.n_cue

    def setup(self):
        _ = self.parameters
        self.n_session = _["n_session"]
        self.nb_train = _["nb_train"]
        self.n_cue = _["n_cue"]
        self.n_pos = _["n_position"]
        self.n_input_time_1 = _["n_input_time_1"]
        self.n_input_time_2 = _["n_input_time_2"]
        self.n_init_time = _["n_init_time"]
        self.n_end_time = _["n_end_time"]
        self.n_input_delay = _["n_input_delay"]
        self.n_choice = _["n_choice"]
        self.reward_probabilities = _["reward_probabilities"]
        self.best_trial_first = None
        self.trials = np.array([*boards(self.n_cue, self.n_pos, self.n_choice)])
        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay

        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

        self.trial_parameter_options_overlap = []
        self.trial_parameter_options_no_overlap = []
        self.trial_parameter_all = []
        overlap_diff_in_input_time = []
        no_overlap_diff_in_input_time = []
        both_diff_in_input_time = []

        for input_time_1 in range(5, 20):
            for delay in range(21):
                for input_time_2 in range(5, 20):
                    if ((self.n_init_time + delay + input_time_2 + self.n_end_time) == 30):
                        if (delay < input_time_1):
                            self.trial_parameter_options_overlap.append([input_time_1, delay, input_time_2])
                            overlap_diff_in_input_time.append(np.abs(input_time_1 - input_time_2))
                        elif (np.abs(input_time_1 - input_time_2) <= 10):
                            self.trial_parameter_options_no_overlap.append([input_time_1, delay, input_time_2])
                            no_overlap_diff_in_input_time.append(np.abs(input_time_1 - input_time_2))
                        both_diff_in_input_time.append(np.abs(input_time_1 - input_time_2))
                        self.trial_parameter_all.append([input_time_1, delay, input_time_2])

        self.fixed_delay_trials = {}

    def set_random_timings(self):
        trial_parameters = self.trial_parameter_options_overlap[
            np.random.choice(len(self.trial_parameter_options_overlap))]
        self.n_input_time_1 = trial_parameters[0]
        self.n_input_delay = trial_parameters[1]
        self.n_input_time_2 = trial_parameters[2]

        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def set_trial_delay(self, delay):
        self.n_input_delay = delay
        self.n_init_1 = self.n_init_time
        self.n_init_2 = self.n_init_time + self.n_input_delay
        self.n_input_time_1 = 10
        self.n_input_time_2 = 10

        if self.n_init_1 + self.n_input_time_1 > self.n_init_2 + self.n_input_time_2:
            self.n_end_1 = self.n_end_time
            self.n_end_2 = self.n_init_1 + self.n_input_time_1 + self.n_end_1 - (self.n_init_2 + self.n_input_time_2)
        else:
            self.n_end_2 = self.n_end_time
            self.n_end_1 = self.n_init_2 + self.n_input_time_2 + self.n_end_2 - (self.n_init_1 + self.n_input_time_1)

    def __getitem__(self, index):
        """ Get trial index """
        return self.trials[index]

    def __len__(self):
        """ Get number of trials """

        return len(self.trials)

    def get_random_trials(self, n=1):
        """ Get a random trial """
        if n == 1:
            index = np.random.randint(len(self))
        else:
            index = random.sample(range(0, len(self)), n)
            mask = np.ones(len(self), dtype=bool)
            mask[index] = False
            self.unseen_trials = self.trials[mask, ...]
        return self.trials[index]

    def separate_cues(self, trial):
        indexes = np.where(trial.sum(axis=1) == 1)[0]
        trial_with_best_cue = np.zeros((self.n_cue, self.n_pos))
        trial_with_worst_cue = np.zeros((self.n_cue, self.n_pos))
        trial_with_best_cue[indexes[0]] = trial[indexes[0]]
        trial_with_worst_cue[indexes[1]] = trial[indexes[1]]
        return trial_with_best_cue, trial_with_worst_cue

    def get_best_choice(self, trial, reward=None):
        """Return the best choice for a given trial"""
        best_cue = np.argmax(trial.sum(axis=1) * reward)
        return int(np.where(trial[best_cue] == 1)[0])

    def is_legal_choice(self, trial, choice):
        """Return whether choice is a legal choice for a given
        trial"""
        return trial.sum(axis=0)[choice] == 1

    def get_legal_choices(self, trial):
        """Return all legal choices for a given trial"""
        return np.nonzero(trial.sum(axis=0))[0]

    def get_reward_probability(self, trial, choice):
        """Return reward probability associated with a choice"""
        cue = np.argwhere(trial[:, choice] == 1)[0]
        return self.reward_probabilities[int(cue)]

    def get_reward(self, trial, choice, reward, penalty):
        """Return reward probability associated with a choice"""
        if self.is_legal_choice(trial, choice):
            p_reward = self.get_reward_probability(trial, choice)
            if random.random() < p_reward:
                return reward[np.argwhere(trial[:, choice] == 1)[0][0]]
            else:
                return 0
        else:
            return penalty   ## CORRECT

    def invert_probabilities(self):
        self.reward_probabilities = np.ones(self.n_cue) - self.reward_probabilities

    def chronogram(self):
        schedule_1 = [(self.n_init_1, (0, 0)), (self.n_input_time_1, (1, 1)), (self.n_end_1, (0, 0))]
        schedule_2 = [(self.n_init_2, (0, 0)), (self.n_input_time_2, (1, 1)), (self.n_end_2, (0, 0))]
        return np.concatenate([
            np.interp(np.arange(n), [0, n - 1], [beg, end])
            for (n, (beg, end)) in schedule_1]),  np.concatenate([np.interp(np.arange(n), [0, n - 1], [beg, end])
                                                                  for (n, (beg, end)) in schedule_2])

    def get_trial_with_chronogram(self, trial, separate_pathway):
        trial_with_best_cue, trial_with_worst_cue = self.separate_cues(trial)
        positions_best = [sum(x) for x in zip(*trial_with_best_cue)]
        cues_best = np.sum(trial_with_best_cue, axis=1)
        best_coords = np.concatenate((positions_best, cues_best))
        positions_worst = [sum(x) for x in zip(*trial_with_worst_cue)]
        cues_worst = np.sum(trial_with_worst_cue, axis=1)
        worst_coords = np.concatenate((positions_worst, cues_worst))
        k = random.choice([0, 1])
        l = 1 - k
        if k == 0:
            self.best_trial_first = True
            chrono_1 = self.chronogram()[0]
            chrono_2 = self.chronogram()[1]
        else:
            self.best_trial_first = False
            chrono_1 = self.chronogram()[1]
            chrono_2 = self.chronogram()[0]

        L1 = []
        L2 = []
        for i, v in enumerate(best_coords):
            L1.append(v * chrono_1)
        for i, v in enumerate(worst_coords):
            L2.append(v * chrono_2)

        trial_with_chronogram_best = np.reshape(np.transpose(L1),
                                             (self.n_init_1 + self.n_input_time_1 +
                                              self.n_end_1, 8))
        trial_with_chronogram_worst = np.reshape(np.transpose(L2),
                                             (self.n_init_1 + self.n_input_time_1 +
                                              self.n_end_1, 8))

        if separate_pathway:
            if self.best_trial_first:
                return trial_with_chronogram_best, trial_with_chronogram_worst
            else:
                return trial_with_chronogram_worst, trial_with_chronogram_best
        else:
            if self.best_trial_first:
                return np.concatenate((trial_with_chronogram_best, trial_with_chronogram_worst), axis=1)
            else:
                return np.concatenate((trial_with_chronogram_worst, trial_with_chronogram_best), axis=1)


    def plot_chronogram(self, trial_with_chronogram):

        to_plot = np.transpose(trial_with_chronogram)

        y_labels = ['stim.1', 'stim.2', 'stim.3', 'stim.4', 'pos.1', 'pos.2', 'pos.3', 'pos.4']

        fig, axs = plt.subplots(8, 1, figsize=(8, 8), constrained_layout=True)

        for i, v in enumerate(to_plot):
            ax = axs[i]
            ax.plot(v, lw=1)
            ax.set_ylabel(y_labels[i], fontsize=12)
            ax.set_yticks([])

            if i == (len(to_plot) - 1):
                ax.set_xticks([self.n_init_1, self.n_init_2,
                               self.n_init_2 + self.n_input_time_2 + self.n_end_2])
                ax.set_xticklabels([f'1st cue ({self.n_init_1}) ', f'2nd cue ({self.n_init_2})',
                                    f'End trial ({self.n_init_2 + self.n_input_time_2 + self.n_end_2})'], fontsize=12)
            else:
                ax.set_xticks([])

            ax.axvline(x=self.n_init_1, ymin=0, ymax=16, color='red', lw=1, alpha=0.5)
            ax.axvline(x=self.n_init_2, ymin=0, ymax=16, color='red', lw=1, alpha=0.5)
            ax.axvline(x=self.n_init_2 + self.n_input_time_2 + self.n_end_2,
                       ymin=0, ymax=16, color='red', lw=1, alpha=0.5)

        fig.suptitle('Chronogram of the Input', fontsize=16)
        fig.text(0.5, 0.00, 'Timesteps', ha='center', fontsize=14)
        plt.tight_layout()
        plt.show()


#-----------------------------------------------------------------------------


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    task = Task(filename='json_files/task.json')

    # Set the random timings of each stimuli
    task.set_random_timings()
    trial = task.get_random_trials(1)

    print('trial:')
    print(trial)

    # Trial. The rows correspond to the positions of the stimuli, the columns corresponds to the stimuli identity
    #[[0 1 0 0]
    # [0 0 0 0]
    # [0 0 1 0]
    # [0 0 0 0]]
    # Here, stimuli 2 is on at position 1 and stimuli 2 is on at position 3.

    # Transform trial to chronogram, as it will be fed by the model
    trial_with_chronogram = task.get_trial_with_chronogram(trial)

    # Plot chronogram as it will be fed by the model
    task.plot_chronogram(trial_with_chronogram)



