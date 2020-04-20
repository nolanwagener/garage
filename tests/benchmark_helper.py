"""helper functions for benchmarks."""
import json
import os
from os import path as osp

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def create_json(csvs, trials, seeds, xs, ys, factors, names):
    """Convert multiple algorithms csvs outputs to json format.

    Args:
        csvs (list[list]): A list of list of csvs which contains all csv files
            for each algorithms in the task.
        trials (int): Number of trials in the task.
        seeds (list[int]): A list of positive integers which is used in the
            algorithms
        xs (list[string]): A list of X column names of algorithms csv.
        ys (list[string]): A list of Y column names of algorithms csv.
        factors (list[int]): A list of factor value for each algorithms
        names (list[string]): A list of name of each algorithms

    Returns:
        dict: a dictionary(json) whose values should contain time_steps
            (x-value) and return values (y-value) for each algorithms for each
            trials

    """
    task_result = {}
    for trial in range(trials):
        trial_seed = 'trial_%d' % (trial + 1)
        task_result['seed'] = seeds[trial]
        task_result[trial_seed] = {}

        dfs = (json.loads(pd.read_csv(csv[trial]).to_json()) for csv in csvs)
        task_result[trial_seed] = {
            name: {
                'time_steps': [float(val) * factor for val in df[x].values()],
                'return': df[y]
            }
            for df, x, y, factor, name in zip(dfs, xs, ys, factors, names)
        }
    return task_result


def plot_average_over_trials(csvs, ys, plt_file, env_id, x_label, y_label,
                             names):
    """Plot mean and confidence area of benchmark from csv files of algorithms.

    x-value is step and y-value depends on the parameter ys.
    Calculate mean and std for the y values and draw a line using mean and
    show confidence area using std.

    Step length of every csv data ans ys should be same.

    Args:
        csvs (list[list]): A list of list of csvs which contains all csv files
            for each algorithms in the task.
        ys (list[int]): A list of Y column names of algorithms csv.
        plt_file (string): Path of the plot png file.
        env_id (string): String contains the id of the environment. (for title)
        x_label (string): label for x axis of the plot
        y_label (string): label for y axis of the plot
        names (list[string]): labels for each line in the graph

    """
    assert all(len(x) == len(csvs[0]) for x in csvs)

    for trials, y, name in zip(csvs, ys, names):
        y_vals = np.array([np.array(pd.read_csv(t)[y]) for t in trials])
        y_mean, y_std = y_vals.mean(axis=0), y_vals.std(axis=0)

        # pylint: disable=unsubscriptable-object
        plt.plot(list(range(y_vals.shape[-1])), y_mean, label=name)
        # pylint: disable=unsubscriptable-object
        plt.fill_between(list(range(y_vals.shape[-1])), (y_mean - y_std),
                         (y_mean + y_std),
                         alpha=.1)

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()


def plot(g_csvs, b_csvs, g_x, g_y, g_z, b_x, b_y, b_z, trials, seeds, plt_file,
         env_id, x_label, y_label):
    """
    Plot benchmark from csv files of garage and baselines.

    :param b_csvs: A list contains all csv files in the task.
    :param g_csvs: A list contains all csv files in the task.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param b_x: X column names of baselines csv.
    :param b_y: Y column names of baselines csv.
    :param trials: Number of trials in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    assert len(b_csvs) == len(g_csvs)
    for trial in range(trials):
        seed = seeds[trial]

        df_g = pd.read_csv(g_csvs[trial])
        df_b = pd.read_csv(b_csvs[trial])

        plt.plot(df_g[g_x],
                 df_g[g_y],
                 label='%s_trial%d_seed%d' % (g_z, trial + 1, seed))
        plt.plot(df_b[b_x],
                 df_b[b_y],
                 label='%s_trial%d_seed%d' % (b_z, trial + 1, seed))

    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(env_id)

    plt.savefig(plt_file)
    plt.close()


def relplot(g_csvs, b_csvs, g_x, g_y, g_z, b_x, b_y, b_z, trials, seeds,
            plt_file, env_id, x_label, y_label):
    """
    Plot benchmark from csv files of garage from multiple trials using Seaborn.

    :param g_csvs: A list contains all csv files in the task.
    :param b_csvs: A list contains all csv files in the task. Pass
        empty list or None if only plotting garage.
    :param g_x: X column names of garage csv.
    :param g_y: Y column names of garage csv.
    :param b_x: X column names of baselines csv.
    :param b_y: Y column names of baselines csv.
    :param trials: Number of trials in the task.
    :param seeds: A list contains all the seeds in the task.
    :param plt_file: Path of the plot png file.
    :param env_id: String contains the id of the environment.
    :return:
    """
    df_g = [pd.read_csv(g) for g in g_csvs]
    df_gs = pd.concat(df_g, axis=0)
    df_gs['Type'] = g_z
    data = df_gs

    if b_csvs:
        assert len(b_csvs) == len(g_csvs)
        df_b = [pd.read_csv(b) for b in b_csvs]
        df_bs = pd.concat(df_b, axis=0)
        df_bs['Type'] = b_z
        df_bs = df_bs.rename(columns={b_x: g_x, b_y: g_y})
        data = pd.concat([df_gs, df_bs])

    ax = sns.relplot(x=g_x, y=g_y, hue='Type', kind='line', data=data)
    ax.axes.flatten()[0].set_title(env_id)

    plt.savefig(plt_file)

    plt.close()


def create_json_file(b_csvs, g_csvs, trails, seeds, b_x, b_y, g_x, g_y,
                     factor_g, factor_b):
    """Convert garage and benchmark csv outputs to json format."""
    task_result = {}
    for trail in range(trails):
        g_res, b_res = {}, {}
        trail_seed = 'trail_%d' % (trail + 1)
        task_result['seed'] = seeds[trail]
        task_result[trail_seed] = {}
        df_g = json.loads(pd.read_csv(g_csvs[trail]).to_json())
        df_b = json.loads(pd.read_csv(b_csvs[trail]).to_json())

        g_res['time_steps'] = list(
            map(lambda x: float(x) * factor_g, df_g[g_x].values()))
        g_res['return'] = df_g[g_y]

        b_res['time_steps'] = list(
            map(lambda x: float(x) * factor_b, df_b[b_x].values()))
        b_res['return'] = df_b[b_y]

        task_result[trail_seed]['garage'] = g_res
        task_result[trail_seed]['baselines'] = b_res
    return task_result


def write_file(result_json, algo):
    """Create new progress.json or append to existing one."""
    latest_dir = './latest_results'
    latest_result = latest_dir + '/progress.json'
    res = {}
    if osp.exists(latest_result):
        res = json.loads(open(latest_result, 'r').read())
    elif not osp.exists(latest_dir):
        os.makedirs(latest_dir)
    res[algo] = result_json
    result_file = open(latest_result, 'w')
    result_file.write(json.dumps(res))
