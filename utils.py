import matplotlib.pyplot as plt
import numpy as np
import os


def nextnonexistent(f):
    """ Check if a file already exist, if yes, creates a new one with the next number at the end.
    Example: input.txt exists, input_1.txt created.
    Input:
    f: text file name
    Output: new file name or same file name."""
    fnew = f
    root, ext = os.path.splitext(f)
    i = 0
    while os.path.exists(fnew):
        i += 1
        fnew = '%s_%i%s' % (root, i, ext)
    return fnew


def save_files(exp, path, testing=False):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    if testing:
        legal_choices_array = exp.legal_choices_array_testing
        success_array = exp.success_array_testing
        success_array_best_first = exp.success_array_best_first_testing
        success_array_best_last = exp.success_array_best_last_testing
        record_output_activity = exp.model.record_output_activity
        all_trials = exp.all_trials_testing

    else:
        legal_choices_array = exp.legal_choices_array
        success_array = exp.success_array
        success_array_best_first = exp.success_array_best_first
        success_array_best_last = exp.success_array_best_last
        record_output_activity = exp.model.record_output_activity
        all_trials = exp.all_trials

    np.save(arr=legal_choices_array,
            file=nextnonexistent(
                path + 'legal_choice_array.npy'))

    np.save(arr=success_array,
            file=nextnonexistent(
                path + 'overall_success_array.npy'))

    np.save(arr=success_array_best_first,
            file=nextnonexistent(
                path + 'best_first_array.npy'))

    np.save(arr=success_array_best_last,
            file=nextnonexistent(
                path + 'best_last_array.npy'))

    np.save(arr=record_output_activity,
            file=nextnonexistent(path + 'output_activity.npy'))

    np.save(arr=exp.model.record_reservoir_states,
            file=nextnonexistent(path + 'reservoir_activity.npy'))

    for name in ('trials', 'delay', 'input_time_1', 'input_time_2'):
        np.save(arr=all_trials[name],
                file=nextnonexistent(path + name + '.npy'))

    #np.save(arr=all_W_out,
    #       file=nextnonexistent(specific_path + 'all_W_out' + '.npy'))


def json_exists(file_name):
    return os.path.exists(file_name)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)*100
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_success(success_array, save=False, folder_to_save=None, name='succes_rate', show=True, title='Percentage of success'):
    """
    Plot the evolution of training of one model. y-axis: success percentage. x-axis: trial number/50.
        parameters:
            success_array: numpy array
                            contains the percentage of success every 50 time steps
            save: boolean
            folder_to_save: str
            name : str
    """
    plt.subplots(figsize=(10, 5))
    plt.plot(success_array, color='black')
    plt.title(title)
    plt.xlabel('Trial number')
    plt.ylabel(title + ' with avg filter')
    plt.ylim((15, 100))
    if save:
        isExist = os.path.exists(folder_to_save)
        if not isExist:
            os.makedirs(folder_to_save)
        plt.savefig(nextnonexistent(folder_to_save + name + '.pdf'))
    if show:
        plt.show()


def plot_output_activity(output_array, n, save=False, folder_to_save=None, show=True,
                         title='Output neuron activity'):

    for i in range(len(output_array)-n, len(output_array)):
        plt.subplots(figsize=(10, 5))
        trial_info = output_array[i]['trial_info']
        output_array[i]['output_activity'] = np.transpose(output_array[i]['output_activity'])
        for k in range(4):
            #print(np.shape(output_array[i]['output_activity'][k][:]))
            if trial_info[k] == 0 or trial_info[k] == -0.01:
                plt.plot(output_array[i]['output_activity'][k][:], alpha=0.2)
            else:
                plt.plot(output_array[i]['output_activity'][k][:], alpha=1)

        plt.legend(['Output 0, r = {}'.format(trial_info[0]), 'Output 1, r = {}'.format(trial_info[1]),
                    'Output 2, r = {}'.format(trial_info[2]), 'Output 3, r = {}'.format(trial_info[3])],
                   bbox_to_anchor=(1., 0.5))

        if trial_info['best_cue_first']:
            plt.title(title + ' when best cue appears first')
        else:
            plt.title(title + ' when best cue appears last')
        plt.xlabel('Time step')
        plt.ylabel(title)
        plt.subplots_adjust(right=0.75)
        if save:
            isExist = os.path.exists(folder_to_save + 'output_activity_fig/')
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(folder_to_save + 'output_activity_fig/')
                print("The new directory is created!")
            plt.savefig(nextnonexistent(folder_to_save + 'output_activity_fig/' + str(i) + '.pdf'))
        if show:
            plt.show()


def plot_w_out(w_out_array, save=False, folder_to_save=None, name='w_out', show=True,
                         title='Output weight value'):

    fig, axs = plt.subplots(nrows=4, figsize=(10, 5))
    w_out_array=np.transpose(w_out_array)

    for i in range(4):
        sub_w = np.transpose(w_out_array[i])

        #axs[i].plot(sub_w[1:,:50])
        axs[i].plot(sub_w[:, :50])
    plt.title(title)
    plt.xlabel('Trial number')
    plt.ylabel(title)
    if save:
        plt.savefig(folder_to_save + name + '.pdf')
    if show:
        plt.show()

def input_connect_dual(X, n_input_cells, n_input, domain):
    norm_X = X/np.max(X)
    cells_in_0 = [[] for i in range(int(n_input/2))]
    cells_in_1 = [[] for i in range(int(n_input/2))]
    total_cells_in = set()
    for j in range(int(n_input/2)):
        for i in np.argsort(X)[:125]:
            if (np.random.uniform(0, 1) < np.exp((-(norm_X[i])**2)/0.014)):
                if domain[i] == 0:
                    cells_in_0[j].append(i)
                    total_cells_in.add(i)
    for j in range(int(n_input/2)):
        for i in np.argsort(X)[:125]:
            if (np.random.uniform(0, 1) < np.exp((-(norm_X[i]) ** 2) / 0.014)):
                if domain[i] == 1:
                    cells_in_1[j].append(i)
                    total_cells_in.add(i)
    # print('Mean Number of Input:',np.mean([len(neuron_list) for neuron_list in cells_in]))
    # print('Unique Number:',len(set(sum(cells_in,[]))))
    return cells_in_0, cells_in_1, total_cells_in

def output_connect(X, n_output_cells, n_output, output_param):

    norm_X = X/np.max(X)
    cells_out = [[] for i in range(n_output)]
    for j in range(n_output):
        for i in np.argsort(X):
            if (np.random.uniform(0, 1) < np.exp((-(1-norm_X[i])**2)/output_param)):
                cells_out[j].append(i)

    # print('Mean Number of Output:',np.mean([len(neuron_list) for neuron_list in cells_out]))
    # print('Unique Number:',len(set(sum(cells_out,[]))))

    return cells_out

def input_connect(X, n_input_cells, n_input):
    norm_X = X/np.max(X)
    cells_in = [[] for i in range(n_input)]
    total_cells_in = set()
    for j in range(n_input):
        for i in np.argsort(X)[:250]:
            if (np.random.uniform(0, 1) < np.exp((-(norm_X[i])**2)/0.014)):
                cells_in[j].append(i)
                total_cells_in.add(i)
    # print('Mean Number of Input:',np.mean([len(neuron_list) for neuron_list in cells_in]))
    # print('Unique Number:',len(set(sum(cells_in,[]))))
    return cells_in, total_cells_in

def connect_setup(P):
    n = len(P)
    dP = P.reshape(1, n, 2) - P.reshape(n, 1, 2)  # calculate all possible vectors
    # Distances
    D = np.hypot(dP[..., 0], dP[..., 1])/1000.0  # calculate the norms of all vectors
    # k nearest neighbours
    A = np.zeros((n, n))
    for i in range(n):
        A[i] = np.arctan2(dP[i, :, 1], dP[i, :, 0]) * 180.0 / np.pi  # angles (arctan2 signed angle, A en degrés)
    n = len(P)
    I = np.argsort(D, axis=1)
    domain = np.zeros(n)
    Y = P[:, 1] #Y-coordinates
    top_neurons = np.argsort(Y)[int(n/2):]
    domain[top_neurons] = 1
    return n, D, A, I, domain

def limit_based_connect(P, total_cells_in, connect_limit, connect_prob, angle):
    """
    Build a connection matrix
    """
    n, D, A, I, domain = connect_setup(P)
    W = np.zeros((n, n))
    mean_connect = 0

    for i in range(n):
        R = D[i]
        p = 0
        p_n = 0
        for j in range(1, n):
            if domain[i] == domain[I[i,j]]:
                if A[i, I[i, j]] > angle[int(domain[i])] or A[i, I[i, j]] < -angle[int(domain[i])]: # connected only from behind
                    if np.random.uniform(0, 1) < connect_prob[int(domain[i])]:
                        W[I[i, j], i] = 1
                        p_n += 1
                    p += 1
                if p > connect_limit[int(domain[i])]:
                    break
        if p_n==0:
            if not(i in total_cells_in):
                W[I[i,1],i] = 1
                p_n += 1

        mean_connect += p_n

    mean_connect = mean_connect/n
    return W, domain

def limit_based_connect_forward(P, total_cells_in, connect_limit, connect_prob, angle):
    """
    Build a connection matrix
    """
    n, D, A, I, domain = connect_setup(P)
    W = np.zeros((n, n))
    mean_connect = 0
    for i in range(n):
        R = D[i]
        p = 0
        p_n = 0
        for j in range(1, n):
            if A[i, I[i, j]] > angle or A[i, I[i, j]] < -angle: # connected only from behind
                if np.random.uniform(0, 1) < connect_prob:
                    W[I[i, j], i] = 1
                    p_n += 1
                p += 1
            if p > connect_limit:
                break
        if p_n==0:
            if not(i in total_cells_in):
                W[I[i,1],i] = 1
                p_n += 1
        mean_connect += p_n
    mean_connect = mean_connect/n
    return W

def build_forward(limit, connect_prob, angle, weight_scale, output_param, seed):
    """
    Parameters:
    -----------
    n_cells:        Number of cells in the reservoir
    n_input_cells:  Number of cells receiving external input
    n_output_cells: Number of cells sending external output
    n_input:        Number of external input
    n_output:       Number of external output
    sparsity:       Connection rate
    seed:           Seed for the random genrator
    """
    np.random.seed(seed)
    n_input = 16
    n_output = 4
    n_cells = 500
    n_input_cells = 50
    n_output_cells = 250
    connect_limit = n_cells * limit
    filename = 'uniform-1024x1024-stipple-500.npy'
    if not os.path.exists(filename):
        x_error = input('File Does not Exist?')
    cells_pos = np.load(filename)
    X, Y = cells_pos[:, 0], cells_pos[:, 1]
    cells_in, total_cells_in = input_connect(X, n_input_cells, n_input)
    cells_out = output_connect(X, n_output_cells, n_output, output_param)
    W = limit_based_connect_forward(cells_pos, total_cells_in, connect_limit, connect_prob, angle)

    W_in = np.zeros((n_input, n_cells))
    for i in range(n_input):
        W_in[i, cells_in[i]] = 1

    W_out = np.zeros((n_cells, n_output))
    for i in range(n_output):
        W_out[cells_out[i], i] = 1

    W_in *= np.random.uniform(-1.0, 1.0, W_in.shape)
    W *= np.random.uniform(-1.0, 1.0, W.shape)
    W_out *= np.random.uniform(-1.0, 1.0, W_out.shape)
    return cells_pos / 1000, np.transpose(W), np.transpose(W_in), W_out

def build(limit, connect_prob, angle, weight_scale, output_param, seed):
    """
    Parameters:
    -----------
    n_cells:        Number of cells in the reservoir
    n_input_cells:  Number of cells receiving external input
    n_output_cells: Number of cells sending external output
    n_input:        Number of external input
    n_output:       Number of external output
    sparsity:       Connection rate
    seed:           Seed for the random genrator
    """
    np.random.seed(seed)
    #n_input = 16
    n_input = 8
    n_output = 4
    n_cells = 500
    n_input_cells = 50
    n_output_cells = 250
    #print(limit)
    # connect_limit = limit*n_cells
    connect_limit = [n_cells * elem for elem in limit]
    # print(connect_limit)
    filename = 'uniform-1024x1024-stipple-500.npy'
    if not os.path.exists(filename):
        x_error = input('File Does not Exist?')
    cells_pos = np.load(filename)
    X, Y = cells_pos[:, 0], cells_pos[:, 1]
    cells_in, total_cells_in = input_connect(X, n_input_cells, n_input)
    cells_out = output_connect(X, n_output_cells, n_output, output_param)
    W, domain = limit_based_connect(cells_pos, total_cells_in, connect_limit, connect_prob, angle)

    W_in = np.zeros((n_input, n_cells))
    for i in range(n_input):
        W_in[i, cells_in[i]] = 1

    W_out = np.zeros((n_cells, n_output))
    for i in range(n_output):
        W_out[cells_out[i], i] = 1

    W_in *= np.random.uniform(-1.0, 1.0, W_in.shape)
    W *= np.random.uniform(-1.0, 1.0, W.shape)
    # if(np.abs(np.linalg.eig(W)[0]).max()>=1):
    #     W     *= spec_rad / np.abs(np.linalg.eig(W)[0]).max()
    # for i in range(len(domain)):
    #     if(domain[i]==1):
    #         W[:,i] *= weight_scale
    # print(weight_scale)
    # W *= weight_scale
    W_out *= np.random.uniform(-1.0, 1.0, W_out.shape)
    #print(domain)

    return cells_pos / 1000, np.transpose(W), np.transpose(W_in), W_out, domain

def build_separate_input(limit, connect_prob, angle, weight_scale, output_param, seed):
    """
    Parameters:
    -----------
    n_cells:        Number of cells in the reservoir
    n_input_cells:  Number of cells receiving external input
    n_output_cells: Number of cells sending external output
    n_input:        Number of external input
    n_output:       Number of external output
    sparsity:       Connection rate
    seed:           Seed for the random genrator
    """
    np.random.seed(seed)
    n_input = 16
    n_output = 4
    n_cells = 500
    n_input_cells = 50
    n_output_cells = 250
    connect_limit = [n_cells * elem for elem in limit]
    filename = 'uniform-1024x1024-stipple-500.npy'
    if not os.path.exists(filename):
        x_error = input('File Does not Exist?')
    cells_pos = np.load(filename)
    n, D, A, I, domain = connect_setup(cells_pos)
    X, Y = cells_pos[:, 0], cells_pos[:, 1]
    cells_in_0, cells_in_1, total_cells_in = input_connect_dual(X, n_input_cells, n_input, domain)
    cells_out = output_connect(X, n_output_cells, n_output, output_param)
    W, domain = limit_based_connect(cells_pos, total_cells_in, connect_limit, connect_prob, angle)

    W_in = np.zeros((n_input, n_cells))
    for i in range(int(n_input/2)):
        W_in[i, cells_in_0[i]] = 1

    for i in range(int(n_input/2)):
        W_in[i+int(n_input/2), cells_in_1[i]] = 1

    W_out = np.zeros((n_cells, n_output))
    for i in range(n_output):
        W_out[cells_out[i], i] = 1
    W_in *= np.random.uniform(-1.0, 1.0, W_in.shape)
    W *= np.random.uniform(-1.0, 1.0, W.shape)
    W_out *= np.random.uniform(-1.0, 1.0, W_out.shape)
    return cells_pos / 1000, np.transpose(W), np.transpose(W_in), W_out, domain

