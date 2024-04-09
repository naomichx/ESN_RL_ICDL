import numpy as np
from utils import plot_success, plot_output_activity, plot_w_out, moving_average#,# plot_several_results
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats
from utils import nextnonexistent
import os
cmap = plt.get_cmap("tab20b", 20)
import matplotlib
matplotlib.rcParams.update({'font.size': 13})
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec

#Options
params = {'text.usetex' : False,
          'font.size' : 15,
           'xtick.labelsize':15,
           'ytick.labelsize':15,
          }
plt.rcParams.update(params)
model_names = {}
model_names['M_0'] = ' $M^{0}$'
model_names['M_1'] = '$M^{1}$'
model_names['M_2'] = '$M^{2}$'
model_names['M_3'] = '$M^{3}$'
model_names['M_star'] = '$M^{*}$'
model_names['M_plus'] = '$M^{+}$'


model_types = ('M_plus', 'M_0', 'M_1', 'M_2', 'M_3', 'M_star')

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def addlabels(ax,x,y):
    for i in range(len(x)):
        ax.text(i - 0.1, y[i] + 15, y[i], ha='center', fontsize=13)


def plot_results_nicolas(ax, Y, names, xlabel, legend=True, yaxis=True, save=False):
    barwidth = 1
    X = barwidth * np.arange(len(Y))
    Y_mean = [np.mean(y) for y in Y]
    Y_std = [np.std(y) for y in Y]
    C = [cmap(2 + i * 4) for i in range(len(Y))]

    ax.errorbar(X, Y_mean, xerr=0, yerr=Y_std, fmt=".", color="black", capsize=0, capthick=2,  elinewidth=2)
    ax.bar(X, height=Y_mean, width=0.9 * barwidth, color=C, edgecolor="white")
    legend_elements = [Line2D([0], [0], color=C[i], label=names[i],
                              markerfacecolor='black', markersize=20) for i in range(len(X))]
    for i in range(len(X)):
        if legend:
            #ax.text(0.1, 1.1 - i * 0.1, names[i], ha="left", va="top",
             #       transform=ax.transAxes, color=C[i], fontsize='x-large')
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.5, 1))
        ax.scatter(X[i] + np.random.normal(0, barwidth / 8, Y[i].size),
                   Y[i], s=15, facecolor="red",
                   edgecolor="white")
    ax.set_ylim(0, 100)

    if not yaxis:
        ax.set_yticks([])
        ax.spines['left'].set_color('white')
        ax.spines['left'].set_visible(True)

    else:
        ax.set_ylabel('% of right choice', labelpad=-5, fontsize="large")
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels([0, 25, 50, 75, 100])
        ax.spines['left'].set_position(('data', -barwidth / 2 - 0.1))
        ax.spines['left'].set_visible(True)

    addlabels(ax, names, [round(num, 1) for num in Y_mean])
    #ax.grid(axis='y')
    print(xlabel)
    ax.set_xlabel(xlabel, labelpad=10, fontsize="large")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['bottom'].set_position(('data', -1))


def plot_training(model_types, model_names,n_seeds_init, n_seeds_end,n_out,
                        avg_filter=50, save=False):
    labels = []
    paths = []
    name = ''
    path = 'results/training/'

    for model_type in model_types:
        paths.append(path + model_type + '/')
        labels.append(model_names[model_type])
        name += '_' + model_type
    res_success = []
    res_best_first = []
    res_best_last = []
    mean_success = {}
    mean_first = {}
    mean_last = {}
    for k in range(len(paths)):
        success_arrays = []
        best_first_arrays = []
        best_last_arrays = []
        for i in range(n_seeds_init, n_seeds_end):
            if i == n_out:
                pass
            else:
                success_arrays.append(np.load(paths[k] +  f'{str(i)}/overall_success_array.npy', allow_pickle=True))
                best_first_arrays.append(np.load(paths[k] + f'{str(i)}/best_first_array.npy', allow_pickle=True))
                best_last_arrays.append(np.load(paths[k] + f'{str(i)}/best_last_array.npy', allow_pickle=True))
        mean_success[k] = np.mean(np.array(success_arrays), axis=0)
        mean_first[k], error = tolerant_mean(np.array(best_first_arrays))
        mean_last[k], error = tolerant_mean(np.array(best_last_arrays))
    for model in range(len(paths)):
        res_success.append(mean_success[model])
        res_best_first.append(mean_first[model])
        res_best_last.append(mean_last[model])

    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle('Mean Training Performance Over 10 Seeds')
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.5,0.5], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])
    C = plt.get_cmap("Reds")(np.linspace(0.1, 1, 6))
    #C = [cmap(2 + i * 4) for i in range(len(res_success))]
    for i in range(len(res_success)):
        res = moving_average(res_success[i], avg_filter)
        ax1.plot(res, color=C[i], label=labels[i])
    #ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax1.set_title('Overall performances')
    ax1.set_ylim((15, 100))
    ax1.set_xlabel('Trial number')
    ax1.set_ylabel('% of right choice')
    ax1.legend(loc='center left', bbox_to_anchor=(0, 1.3),ncol=6)

    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(len(res_best_first)):
        res = moving_average(res_best_first[i], avg_filter)
        ax2.plot(res, color=C[i], label=labels[i])
    ax2.set_ylabel('% of right choice')
    ax2.set_title('Best first')
    ax2.set_ylim((15, 100))
    ax2.set_xlabel('Trial number')

    ax3 = fig.add_subplot(gs[1, 1],sharey=ax2)
    for i in range(len(res_best_last)):
        res = moving_average(res_best_last[i], avg_filter)
        ax3.plot(res, color=C[i], label=labels[i])
    #ax3.set_ylabel('% of success with average filter n={}'.format(avg_filter))
    ax3.set_title('Best last')
    ax3.set_ylim((15, 100))
    ax3.set_xlabel('Trial number')
    fig.tight_layout()
    if save:
        print(path + str(n_seeds_init) + '-' + str(n_seeds_end) + '/' + 'all_train_final.pdf')
        plt.savefig(nextnonexistent(path + str(n_seeds_init) + '-' + str(n_seeds_end) + '/' + 'all_train_final.pdf'))

    plt.show()


def plot_testing(model_types, model_names, n_seeds_init, n_seeds_end, show=False, save=False):
    labels = []
    paths = []
    name = ''
    path = 'results/testing/'
    for model_type in model_types:
        paths.append(path + model_type + '/')
        labels.append(model_names[model_type])
        name += '_' + model_type

    sub_folders = model_types
    labels = ['Overall', 'Best first', 'Best last']
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(len(sub_folders) * 1.5, 7),
                             sharey=False, sharex=True)

    #axes[-1, -1].axis('off')
    for s, model_type in enumerate(model_types):
        Y = []
        success_arrays = []
        best_first_arrays = []
        best_last_arrays = []
        testing_path = path + model_type

        for i in range(n_seeds_init, n_seeds_end):
            success_arrays.append(
                np.mean(np.load(testing_path + f'/{str(i)}/overall_success_array.npy',
                                allow_pickle=True)) * 100)
            best_first_arrays.append(np.mean(
                np.load(testing_path+  f'/{str(i)}/best_first_array.npy',
                        allow_pickle=True)) * 100)
            best_last_arrays.append(
                np.mean(np.load(testing_path + f'/{str(i)}/best_last_array.npy',
                                allow_pickle=True)) * 100)
        if s < 3:
            ax = axes[0, s]
        else:
            ax = axes[1, int(s-3)]
        #if model_type != 'M_0' or model_type != 'M_1':
            #ttest_overall = round(compute_t_test('regular_separate_input', model_type, n_seeds_init, n_seeds_end, n_out)['success'][1], 8)
            #ttest_first = round(compute_t_test('regular_separate_input', model_type, n_seeds_init, n_seeds_end, n_out)['best_first'][1], 8)
            #ttest_last = round(compute_t_test('regular_separate_input', model_type, n_seeds_init, n_seeds_end, n_out)['best_last'][1], 8)
            #print(model_type)
            #print('overall', ttest_overall)
            #print('first', ttest_first)
            #print('last', ttest_last)

        Y.append(success_arrays)
        Y.append(best_first_arrays)
        Y.append(best_last_arrays)
        Y = np.array(Y)
        title = 'Mean Testing Performance Over 10 Seeds'
        if s==0 or s == 3:
            plot_results_nicolas(ax, Y, labels, model_names[model_type], legend=False, yaxis=True)

        else:
            plot_results_nicolas(ax, Y, labels, model_names[model_type], legend=False, yaxis=False)

    C = [cmap(2 + i * 4) for i in range(len(Y))]
    legend_elements = [Line2D([0], [0], color=C[i], label=labels[i],
                              markerfacecolor='black', markersize=20) for i in range(len(Y))]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1.0))

    fig.suptitle(title)
    fig.subplots_adjust(hspace=.0)
    plt.tight_layout()

    isExist = os.path.exists(path + str(n_seeds_init) + '-' + str(n_seeds_end) + '/')
    if not isExist:
        os.makedirs(path + str(n_seeds_init) + '-' + str(n_seeds_end) + '/')
    if save:
        plt.savefig(nextnonexistent(path + str(n_seeds_init) + '-' + str(n_seeds_end) + '/'  + 'all_perf.pdf'))
    if show:
        plt.show()


if __name__ == '__main__':
    plot_training(model_types, model_names, n_seeds_init=0, n_seeds_end=9, n_out=None)
    plot_testing(model_types, model_names, n_seeds_init=0, n_seeds_end=9, show=True, save=False)

