import warnings
import numpy as np
from experiment import Experiment
from utils import plot_success, moving_average, plot_output_activity, plot_w_out, \
    nextnonexistent, json_exists, save_files


model_type = 'M_0_bis'
testing = True
show_plots = False
save = True
n_seed = 11


if __name__ == '__main__':
    avg_filtering = 50
    model_file = 'json_files/' + f'{model_type}.json'
    task_file = 'json_files/task.json'
    for i in range(n_seed):
        # Initialize experiment
        exp = Experiment(model_file=model_file, task_file=task_file, model_type=model_type, seed=i, testing=testing,
                         i_sim=i)

        # Run experiment
        exp.run()

        # Compute moving averages of success rates
        res = moving_average(exp.success_array, avg_filtering)
        res_best_first = moving_average(exp.success_array_best_first, avg_filtering)
        res_best_last = moving_average(exp.success_array_best_last, avg_filtering)
        res_legal_choices = moving_average(exp.legal_choices_array, avg_filtering)

        print('Mean success on testing set: ', np.mean(exp.success_array_testing)*100)

        # Plot success rates
        plot_params = {'show': show_plots, 'save': save, 'folder_to_save': f'results/training/{model_type}/'+str(i)+ '/'}
        plot_success(res, **plot_params, name='success_rate')
        plot_success(res_best_first, **plot_params, name='best_first_success_rate')
        plot_success(res_best_last, **plot_params, name='best_last_success_rate')
        plot_success(res_legal_choices, **plot_params, name='legal_choice')

        if save:
            print('Saving all npy files in ', 'results/', '...')
            save_files(exp,  f'results/training/{model_type}/' + str(i) + '/')
            if testing:
                save_files(exp,  f'results/testing/{model_type}/' + str(i) + '/', testing=testing)


