"""

Hyperparameter optimization of the ESN-RL model using the Optuna and Mlflow libraries.

"""
import os
import json
import numpy as np
import optuna
from typing import Dict
import mlflow
from experiment import Experiment
from utils import moving_average
import random
SEED = 1
random.seed(SEED)
np.random.seed(SEED)

def save_to_disk(path, agent_id, hparams, nrmse):
    try:
        # Create target Directory
        os.mkdir(path + '/' + str(agent_id) + '/')
        print("Directory ", path + '/' + str(agent_id) + '/', " Created ")
    except FileExistsError:
        print("Directory ", path + '/' + str(agent_id) + '/', " already exists")
    with open(path + '/' + str(agent_id) + '/' + 'hparams.json', 'w') as f:
        json.dump(hparams, f)
    np.save(path + '/' + str(agent_id) + '/' + 'nrmse.npy', nrmse)


def get_agent_id(agend_dir_1):
    try:
        os.mkdir(agend_dir_1)
        print("Directory ", agend_dir_1, " Created ")
    except FileExistsError:
        print("Directory ", agend_dir_1, " already exists")
    ids = []
    for id in os.listdir(agend_dir_1):
        try:
            ids.append(int(id))
        except:
            pass
    if ids == []:
        agent_id = 1
    else:
        agent_id = max(ids) + 1
    return str(agent_id)


def sample_hyper_parameters(trial: optuna.trial.Trial, n_reservoir) -> Dict:
    sr = {}
    lr = {}
    rc_connectivity = {}
    input_connectivity = {}
    fb_connectivity = {}
    n_units = {}
    fb_scaling = {}
    for i in range(n_reservoir):
        sr[i] = trial.suggest_loguniform("sr_{}".format(str(i)), 1e-2, 1)
        lr[i] = trial.suggest_loguniform("lr_{}".format(str(i)), 1e-4, 1)
        input_connectivity[i] = trial.suggest_loguniform("input_connectivity_{}".format(str(i)),  0.1, 1)
        #trial.suggest_float(input_connectivity_{}".format(str(i)),  0.1, 1)
        rc_connectivity[i] = trial.suggest_loguniform("rc_connectivity_{}".format(str(i)), 1e-4, 0.9)
        #rc_connectivity[i] = 0.05
        #fb_connectivity[i] = trial.suggest_loguniform("fb_connectivity_{}".format(str(i)), 1e-4, 1)
        #fb_connectivity[i] = trial.suggest_loguniform("fb_connectivity_{}".format(str(i)), 1e-2, 1)
        fb_connectivity[i] = 0.1
        #fb_scaling[i] = trial.suggest_loguniform("fb_scaling_{}".format(str(i)), 0.01, 10)
        fb_scaling[i] = 0.01
    #fb_connectivity[0] = 0
    #fb_connectivity[1] = trial.suggest_loguniform("fb_connectivity_{}".format(str(1)), 1e-4, 0.9)
    output_connectivity = trial.suggest_loguniform("output_connectivity", 0.1, 0.99)
    #output_connectivity = 0.8
    dict = {}
    beta = trial.suggest_int("beta", 5, 20)
    eta = trial.suggest_loguniform("eta", 1e-4, 1e-1)
    #eta = trial.suggest_loguniform("eta", 1e-2, 6e-2)
    decay = trial.suggest_float("decay",  0.3, 0.9)
    r_th = 1e-6
    #decay = trial.suggest_loguniform("decay", 0.3, 0.5)
    dict['sr'] = sr
    dict['lr'] = lr
    dict['input_connectivity'] =  input_connectivity
    dict['output_connectivity'] = output_connectivity
    dict['rc_connectivity'] = rc_connectivity
    dict['beta'] = beta
    dict['eta']  = eta
    dict['decay'] = decay
    dict['n_units'] = n_units
    dict['fb_connectivity'] = fb_connectivity
    dict['fb_scaling'] = fb_scaling
    dict['r_th'] = r_th
    return dict


def sample_hyper_parameters_continuous(trial: optuna.trial.Trial, domain ) -> Dict:
    fb_connectivity = 0.1
    fb_scaling = 0.01
    connect_limit = {}
    connect_prob = {}
    lr = {}
    angle = {}
    if domain:
        n = 2
    else:
        n = 1
    for i in range(n):
        lr[i] = trial.suggest_loguniform("lr_{}".format(str(i)), 1e-4, 1)
        connect_limit[i] = trial.suggest_loguniform("connect_limit_{}".format(str(i)),  1e-3, 1)
        connect_prob[i] = trial.suggest_loguniform("connect_prob_{}".format(str(i)),  1e-3, 1)
        angle[i] = trial.suggest_int("angle_{}".format(str(i)),  10, 120)
    output_connectivity = trial.suggest_loguniform("output_connectivity", 0.1, 40)
    sr = trial.suggest_loguniform("sr", 1e-3, 1.5)
    dict = {}
    beta = trial.suggest_int("beta", 5, 20)
    eta = trial.suggest_loguniform("eta", 1e-4, 1e-1)
    decay = trial.suggest_loguniform("decay",  0.1, 0.9)
    r_th = 1e-6
    dict['sr'] = sr
    dict['lr'] = lr
    dict['connect_limit'] =  connect_limit
    dict['connect_prob'] = connect_prob
    dict['angle'] = angle
    dict['output_connectivity'] = output_connectivity
    dict['beta'] = beta
    dict['eta']  = eta
    dict['decay'] = decay
    dict['fb_connectivity'] = fb_connectivity
    dict['fb_scaling'] = fb_scaling
    dict['r_th'] = r_th
    return dict


def objective(trial: optuna.trial.Trial, agent_dir, model_file, task_file, model_type, n_res):
    with mlflow.start_run():
        agent_id = get_agent_id(agent_dir)
        mlflow.log_param('agent_id', agent_id)
        if model_type == 'M_star' or model_type == 'M_0_bis':
            if model_type == 'M_star':
                domain = True
            else:
                domain = False
            arg = sample_hyper_parameters_continuous(trial, domain=domain)
            mlflow.log_params(trial.params)
            exp = Experiment(seed=SEED, model_file=model_file, task_file=task_file, model_type=model_type,
                             hyperparam_optim=True,
                             lr=arg['lr'], sr=arg['sr'], connect_limit=arg['connect_limit'],
                             angle=arg['angle'], connect_prob=arg['connect_prob'],
                             fb_connectivity=arg['fb_connectivity'],
                             output_connectivity=arg['output_connectivity'],
                             eta=arg['eta'], beta=arg['beta'], r_th=arg['r_th'],
                             decay=arg['decay'], i_sim=agent_id)
        else:
            arg = sample_hyper_parameters(trial, n_res)
            mlflow.log_params(trial.params)
            exp = Experiment(seed=SEED, model_file=model_file, task_file=task_file, model_type=model_type,
                             hyperparam_optim=True,
                             lr=arg['lr'], sr=arg['sr'],
                             rc_connectivity=arg['rc_connectivity'], fb_connectivity=arg['fb_connectivity'],
                             output_connectivity=arg['output_connectivity'],
                             input_connectivity=arg['input_connectivity'], eta=arg['eta'], beta=arg['beta'],
                             r_th=arg['r_th'],
                             decay=arg['decay'], i_sim=agent_id)
        exp.run()
        avg = 50
        session_scores = moving_average(exp.success_array, avg)
        #session_scores_best_first = moving_average(exp.success_array_best_first['session 0'], avg)
        #session_scores_best_last = moving_average(exp.success_array_best_last['session 0'], avg)
        #session_scores_legal_choices = moving_average(exp.legal_choices_array['session 0'], avg)

        score = np.mean(session_scores[-50:])
        print('Score: ', score)
        save_to_disk(agent_dir, agent_id, arg, score)
        mlflow.log_metric('percent_success', score)
        return score


def optuna_optim(title, model_type, n_res, n_trials=600):

    model_file = f'json_files/{model_type}.json'
    task_file = 'json_files/task.json'

    print('Start Optuna optimization ...')
    parent_dir = 'optuna_results'
    title = title
    EXPERIMENT_NAME = 'hyperparameter_search_' + title
    SAVED_AGENTS_DIR = parent_dir + '/mlagent/' + title
    MLFLOW_RUNS_DIR = parent_dir + '/mlflows/' + title

    mlflow.set_tracking_uri(MLFLOW_RUNS_DIR)
    mlflow.set_experiment(EXPERIMENT_NAME)

    task_file_path = parent_dir + "/task_params/" + title
    isExist = os.path.exists(task_file_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(task_file_path)
        print("The new directory is created!")

    model_file_path = parent_dir + "/model_params/" + title
    isExist = os.path.exists(model_file_path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(model_file_path)
        print("The new directory is created!")
    with open(task_file, "r") as init:
        with open(task_file_path+"/task.json", "w") as to:
            json_string = json.dumps(json.load(init))
            json.dump(json_string, to)
    with open(model_file, "r") as init:
        with open(model_file_path+"/model.json", "w") as to:
            json_string = json.dumps(json.load(init))
            json.dump(json_string, to)
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(), study_name='optim_' + title,
                                direction='maximize',
                                load_if_exists=True,
                                storage='sqlite:////Users/nchaix/Documents/PhD/code/RL_RC/optuna_results/optuna_db/'
                                        + title + '.db')
    func = lambda trial: objective(trial, agent_dir=SAVED_AGENTS_DIR, model_file=model_file, task_file=task_file,
                                    n_res=n_res, model_type=model_type)
    study.optimize(func, n_trials=n_trials)
    best_trial = study.best_trial
    hparams = {k: best_trial.params[k] for k in best_trial.params if k != 'seed'}
    print(hparams)


if __name__ == '__main__':

    model_type = "M_0_bis"
    n_res = 1
    title = f'{model_type}_' + '26_03'
    optuna_optim(title=title, model_type=model_type, n_res=n_res, n_trials=600)
