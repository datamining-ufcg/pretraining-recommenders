"""
Leakage Analysis

Usage:
    leakage_analysis.py [--plot=<p>]

Options:
    -h --help               Shows this screen.
    --plot=<p>              Should generate plt image files [default: 1]
"""
import os
import json
import pandas as pd

from docopt import docopt
from tqdm import tqdm
from matplotlib import pyplot as plt

INTERESTING_COLS = [
    'dataset', 'dataset_split', 'method', 'last_epoch',
    'rmse', 'n_emb', 'limit', 'init', 'results'
]

MODELS_PATH = '../models'


def update_rmse(row, validation1, validation2):
    if (row['dataset'] in ['ML100k', 'ML1M']):
        last_epoch = validation1.loc[row['dataset'],
                                     row['method'], row['limit'], row['init']]['last_epoch']
    else:
        last_epoch = validation2.loc[row['dataset'],
                                     row['method'], row['limit']]['last_epoch']

    new_rmse = row['results']['val_loss']
    try:
        new_rmse = new_rmse[last_epoch -
                            1] if 'results' in row else row['rmse']
    except IndexError as ie:
        new_rmse = new_rmse[-1]

    row['rmse2'] = new_rmse
    row['last_epoch2'] = last_epoch

    return row


def read_jsons(path, folder_name):
    json_results = []

    for folder in tqdm(os.listdir(f'{path}/{folder_name}')):
        if folder.endswith('.zip') or folder.endswith('.py'):
            continue

        with open(f'{path}/{folder_name}/{folder}/config.json', 'r') as jsf:
            config = json.load(jsf)

        config = {k: v for k, v in config.items() if k in INTERESTING_COLS}
        if 'method' not in config:
            continue

        if (config['method'].startswith('Pretrained')):
            _method = 'Pretrained from ' + config['method'].split('_')[1]
            config['method'] = _method

        if (config['dataset'] == 'ExplicitML100kLeakage'):
            config['dataset'] = 'ExplicitMLTransfer'

        if 'is_complete' in config and not config['is_complete']:
            continue

        json_results.append(config)

    df_results = pd.DataFrame(json_results)
    df_results.drop_duplicates(
        subset=['dataset', 'method', 'limit', 'init', 'dataset_split'], inplace=True)

    return df_results


def transform_results(results):
    sorted_results = results.sort_values(
        by=['dataset', 'dataset_split', 'init'])
    sorted_results.reset_index(drop=True, inplace=True)

    sorted_results.loc[
        sorted_results['method'] == 'Pretrained from ML100kNegativeLeakage', 'method'
    ] = 'Pretrained from ML100kNegativeLeakage_random'

    validation1 = sorted_results.groupby(['dataset', 'method', 'limit', 'init']).agg({
        'last_epoch': 'min'
    })

    validation2 = sorted_results[~sorted_results.dataset.isin(['ML100k', 'ML1M'])].groupby(
        ['dataset', 'method', 'limit']
    ).agg({
        'last_epoch': 'min'
    })

    sorted_results = sorted_results.apply(
        update_rmse, axis=1, validation1=validation1, validation2=validation2)
    sorted_results.drop('results', axis=1, inplace=True)

    return sorted_results


def group_results(to_group):
    grouped = to_group.groupby(['dataset', 'method', 'init', 'limit']).agg(
        {'rmse2': ['mean', 'std', 'count']})
    grouped.reset_index(drop=False, inplace=True)
    grouped = grouped[grouped['limit'].isin([v / 10 for v in range(0, 11)])]
    grouped = grouped[~grouped['init'].isnull()].reset_index(drop=True)

    return grouped


def filter_grouped(to_filter):
    filtered = to_filter.copy()
    filtered['rmse_mean'] = filtered['rmse2']['mean']
    filtered['rmse_std'] = filtered['rmse2']['std']
    filtered['count'] = filtered['rmse2']['count']
    filtered.drop(['rmse2'], axis=1, inplace=True)

    filtered['init'] = pd.Categorical(
        filtered['init'],
        ['random', 'pca', 'word2vec'],
        ordered=True
    )

    return filtered


def fix_grouped_cols(to_fix):
    fixed = to_fix.copy()
    cnames = [c[0] for c in to_fix.columns]
    fixed.columns = cnames
    return fixed


def maybe_create_folder(fname):
    try:
        os.mkdir(f'../results/{fname}')
    except OSError:
        pass


def plot_grouped(grouped, parent_results_folder, fi):
    methods = grouped['method'].unique().tolist()

    colors = {
        'random': '#1b9e77',
        'pca': '#d95f02',
        'word2vec': '#7570b3'
    }

    plt.figure(figsize=(16, 12))
    for i, method in enumerate(methods):
        pt_from = grouped[grouped['method'] == method].reset_index(drop=True)

        plt.subplot(2, 2, i + 1)
        plt.title(f'{method}')

        for inst in pt_from['init'].unique().tolist():

            limit = pt_from[pt_from['init'] == inst]['limit']
            rmse = pt_from[pt_from['init'] == inst]['rmse_mean']

            plt.plot(limit, rmse, color=colors[inst], label=inst)
            plt.scatter(limit, rmse, color=colors[inst])

    plt.legend()
    plt.savefig(f'../results/{parent_results_folder}/{fi}.png')


def eligible_folders(target_folder, folder_names):
    eligible = []
    for fname in folder_names:
        if ('.' in fname):
            continue

        names = os.listdir(f'{MODELS_PATH}/{target_folder}/{fname}')
        names.sort()
        has_source = 'source' in names
        has_target = 'target' in names

        if has_source and has_target:
            eligible.append(fname)

    print(f'Eligible folders: {", ".join(eligible)}')

    return eligible


if __name__ == "__main__":
    arguments = docopt(__doc__, version='Leakage Analysis v0.0.1')

    SHOULD_PLOT = bool(int(arguments['--plot']))
    target_folders = [
        f for f in os.listdir(MODELS_PATH) if f.startswith('target_') and not f.endswith('.zip')
    ]
    target_folders.sort()

    for target_folder in target_folders:
        folders = os.listdir(f'{MODELS_PATH}/{target_folder}')
        parent_folders = eligible_folders(target_folder, folders)
        parent_folders.sort()

        for parent_results_folder in parent_folders:
            maybe_create_folder(target_folder)
            maybe_create_folder(f'{target_folder}/{parent_results_folder}')
            result_path = f'{MODELS_PATH}/{target_folder}/{parent_results_folder}/target'
            result_folders = os.listdir(result_path)
            result_folders.sort()

            for fi in result_folders:
                if fi.endswith('.zip'):
                    continue

                results = read_jsons(result_path, fi)
                results = transform_results(results)
                grouped = group_results(results)
                grouped = filter_grouped(grouped)
                grouped = fix_grouped_cols(grouped)

                grouped.to_csv(
                    f'../results/{target_folder}/{parent_results_folder}/{fi}.csv', index=False)

                if (SHOULD_PLOT):
                    plot_grouped(
                        grouped, f'{target_folder}/{parent_results_folder}', fi)

                print(f'{fi} finished.\n')
