import os
import itertools
from joblib import load

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


def triple_plot_customization(ax, title, xlabel, ylabel, legend=None, xticks=None, lim_min=None, lim_max=None):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(linewidth=0.5)
    if legend is None:
        ax.legend()
    else:
        ax.legend(loc=legend)
    if xticks is not None:
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_xticks(range(len(xticks)))
    if lim_min is not None and lim_max is not None:
        ax.set_ylim(0.99 * lim_min, 1.01 * lim_max)


dataset2label = {
    'gs': 'Serendipity',
    'gs_a': 'Novelty',
    'gs_b': 'Unexpectedness',
    'gs_g': 'Relevance'
}

for dataset in ['gs', 'gs_a', 'gs_b', 'gs_g']:
    PATH = os.path.join('results', dataset)

    models = [
        load(os.path.join(PATH, 'KNNpopularity.joblib')),
        load(os.path.join(PATH, 'MatrixFactorization.joblib')),
        load(os.path.join(PATH, 'AutoRec.joblib')),
        load(os.path.join(PATH, 'ease.joblib'))
    ]

    width = 0.3
    x = np.arange(1, 6)
    k_list = np.array([2, 3, 4, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 150, 300, 500, 1000])
    components = np.array([1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
    layers = [4, 8, 16, 64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
    ease_lambdas = [1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 2000.0, 5000.0, 10000.0]

    ### KNN
    knn_serendipity_ = []
    knn_recall_ = []
    knn_coverage_ = []

    for i in models[0]:
        if i[0]['k'] not in k_list:
            continue
        knn_recall_.append(i[1])
        knn_coverage_.append(i[2])
        knn_serendipity_.append(i[3])

    knn_serendipity = np.array(knn_serendipity_)
    knn_recall = np.array(knn_recall_)
    knn_coverage = np.array(knn_coverage_)

    ### Matrix factorization
    mf_serendipity = []
    mf_recall = []
    mf_coverage = []

    for i in models[1]:
        mf_recall.append(i[1])
        mf_coverage.append(i[2])
        mf_serendipity.append(i[3])

    ### AutoRec
    autorec_serendipity = []
    autorec_recall = []
    autorec_coverage = []

    for i in range(0, len(models[2]) - 1, 3):
        autorec_recall.append(np.mean(np.array(models[2][i:i + 3])[:, 1]))
        autorec_coverage.append(np.mean(np.array(models[2][i:i + 3])[:, 2]))
        autorec_serendipity.append(np.mean(np.array(models[2][i:i + 3])[:, 3]))

    ### EASE
    ease_recall = []
    ease_coverage = []
    ease_serendipity = []

    for i in models[3]:
        ease_recall.append(i[1])
        ease_coverage.append(i[2])
        ease_serendipity.append(i[3])

    recalls = list(itertools.chain(knn_recall, mf_recall, autorec_recall, ease_recall))
    coverages = list(itertools.chain(knn_coverage, mf_coverage, autorec_coverage, ease_coverage))
    serendipities = list(itertools.chain(knn_serendipity, mf_serendipity, autorec_serendipity, ease_serendipity))
    min_recall, max_recall = np.min(recalls), np.max(recalls)
    min_coverage, max_coverage = np.min(coverages), np.max(coverages)
    min_serendipity, max_serendipity = np.min(serendipities), np.max(serendipities)

    font_size = 11
    ser_label = dataset2label[dataset]
    dataset_name = '1m'
    img_folder = 'img'
    img_ext = '.png'
    path_postfix = ''
    if not os.path.exists(os.path.join(PATH, img_folder)):
        os.mkdir(os.path.join(PATH, img_folder))

    fig, ax = plt.subplots(4, 3, constrained_layout=True, figsize=(20, 12))
    fig.suptitle('Models overview')

    plt.title('MovieLens-{}'.format(dataset_name))

    # Models overview
    ax[0, 0].plot(range(len(k_list)), knn_serendipity, 'o-', label=ser_label, color='C0')
    ax[0, 1].plot(range(len(k_list)), knn_recall, 'o-', label='Recall', color='orange')
    ax[0, 2].plot(range(len(k_list)), knn_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[0, 0], 'KNN popular', 'K', 'Metric value', xticks=k_list)
    triple_plot_customization(ax[0, 1], 'KNN popular', 'K', 'Metric value', xticks=k_list)
    triple_plot_customization(ax[0, 2], 'KNN popular', 'K', 'Metric value', xticks=k_list)

    ax[1, 0].plot(range(len(components)), mf_serendipity, 'o-', label=ser_label)
    ax[1, 1].plot(range(len(components)), mf_recall, 'o-', label='Recall', color='orange')
    ax[1, 2].plot(range(len(components)), mf_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[1, 0], 'Matrix factorization', 'Number of components', 'Metric value',
                              legend='lower right', xticks=components)
    triple_plot_customization(ax[1, 1], 'Matrix factorization', 'Number of components', 'Metric value',
                              legend='lower right', xticks=components)
    triple_plot_customization(ax[1, 2], 'Matrix factorization', 'Number of components', 'Metric value',
                              legend='lower right', xticks=components)

    ax[2, 0].plot(range(len(layers)), autorec_serendipity, 'o-', label=ser_label)
    ax[2, 1].plot(range(len(layers)), autorec_recall, 'o-', label='Recall', color='orange')
    ax[2, 2].plot(range(len(layers)), autorec_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[2, 0], 'AutoRec', 'Hide layer size', 'Metric value', xticks=layers)
    triple_plot_customization(ax[2, 1], 'AutoRec', 'Hide layer size', 'Metric value', xticks=layers)
    triple_plot_customization(ax[2, 2], 'AutoRec', 'Hide layer size', 'Metric value', xticks=layers)

    ax[3, 0].plot(range(len(ease_lambdas)), ease_serendipity, 'o-', label=ser_label)
    ax[3, 1].plot(range(len(ease_lambdas)), ease_recall, 'o-', label='Recall', color='orange')
    ax[3, 2].plot(range(len(ease_lambdas)), ease_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[3, 0], 'EASE', 'Lambda', 'Metric value', xticks=ease_lambdas)
    triple_plot_customization(ax[3, 1], 'EASE', 'Lambda', 'Metric value', xticks=ease_lambdas)
    triple_plot_customization(ax[3, 2], 'EASE', 'Lambda', 'Metric value', xticks=ease_lambdas)

    plt.savefig(os.path.join(PATH, img_folder, '3models' + path_postfix + img_ext))

    # Models overview (fixed y)
    fig, ax = plt.subplots(4, 3, constrained_layout=True, figsize=(20, 12))
    fig.suptitle('Models overview (fixed y-axis)')
    plt.title('MovieLens-{}'.format(dataset_name))

    ax[0, 0].plot(range(len(k_list)), knn_serendipity, 'o-', label=ser_label, color='C0')
    # ax[0, 0].plot(range(len(k_list)), knn_b0_serendipity, 'o--', label=ser_label + r', $\beta=0$', color='C0')
    ax[0, 1].plot(range(len(k_list)), knn_recall, 'o-', label='Recall', color='orange')
    # ax[0, 1].plot(range(len(k_list)), knn_b0_recall, 'o--', label=r'Recall, $\beta=0$', color='orange')
    ax[0, 2].plot(range(len(k_list)), knn_coverage, 'o-', label='Coverage', color='green')
    # ax[0, 2].plot(range(len(k_list)), knn_b0_coverage, 'o--', label=r'Coverage, $\beta=0$', color='green')
    triple_plot_customization(ax[0, 0], 'KNN popular', 'K', 'Metric value', xticks=k_list, lim_min=min_serendipity,
                              lim_max=max_serendipity)
    triple_plot_customization(ax[0, 1], 'KNN popular', 'K', 'Metric value', xticks=k_list, lim_min=min_recall,
                              lim_max=max_recall)
    triple_plot_customization(ax[0, 2], 'KNN popular', 'K', 'Metric value', xticks=k_list, lim_min=min_coverage,
                              lim_max=max_coverage)

    ax[1, 0].plot(range(len(components)), mf_serendipity, 'o-', label=ser_label)
    ax[1, 1].plot(range(len(components)), mf_recall, 'o-', label='Recall', color='orange')
    ax[1, 2].plot(range(len(components)), mf_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[1, 0], 'Matrix factorization', 'Number of components', 'Metric value',
                              legend='lower right', xticks=components, lim_min=min_serendipity, lim_max=max_serendipity)
    triple_plot_customization(ax[1, 1], 'Matrix factorization', 'Number of components', 'Metric value',
                              legend='lower right', xticks=components, lim_min=min_recall, lim_max=max_recall)
    triple_plot_customization(ax[1, 2], 'Matrix factorization', 'Number of components', 'Metric value',
                              legend='lower right', xticks=components, lim_min=min_coverage, lim_max=max_coverage)

    ax[2, 0].plot(range(len(layers)), autorec_serendipity, 'o-', label=ser_label)
    ax[2, 1].plot(range(len(layers)), autorec_recall, 'o-', label='Recall', color='orange')
    ax[2, 2].plot(range(len(layers)), autorec_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[2, 0], 'AutoRec', 'Hide layer size', 'Metric value', xticks=layers,
                              lim_min=min_serendipity, lim_max=max_serendipity)
    triple_plot_customization(ax[2, 1], 'AutoRec', 'Hide layer size', 'Metric value', xticks=layers, lim_min=min_recall,
                              lim_max=max_recall)
    triple_plot_customization(ax[2, 2], 'AutoRec', 'Hide layer size', 'Metric value', xticks=layers,
                              lim_min=min_coverage, lim_max=max_coverage)

    ax[3, 0].plot(range(len(ease_lambdas)), ease_serendipity, 'o-', label=ser_label)
    ax[3, 1].plot(range(len(ease_lambdas)), ease_recall, 'o-', label='Recall', color='orange')
    ax[3, 2].plot(range(len(ease_lambdas)), ease_coverage, 'o-', label='Coverage', color='green')
    triple_plot_customization(ax[3, 0], 'EASE', 'Lambda', 'Metric value', xticks=ease_lambdas, lim_min=min_serendipity,
                              lim_max=max_serendipity)
    triple_plot_customization(ax[3, 1], 'EASE', 'Lambda', 'Metric value', xticks=ease_lambdas, legend='lower right',
                              lim_min=min_recall, lim_max=max_recall)
    triple_plot_customization(ax[3, 2], 'EASE', 'Lambda', 'Metric value', xticks=ease_lambdas, lim_min=min_coverage,
                              lim_max=max_coverage)

    plt.savefig(os.path.join(PATH, img_folder, '3models_absolute' + path_postfix + img_ext))

    # Recall vs serendipity
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(16, 12))
    fig.suptitle('Recall vs serendipity')

    plt.title('Recall versus serendipity (MovieLens-{})'.format(dataset_name))

    ax[0, 0].plot(knn_recall, knn_serendipity, 'o-')
    tmp = zip(knn_recall, knn_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 0].text(p[0], p[1], k_list[i], fontsize=font_size, ha='right', va='bottom')

    # ax[0].plot(knn_b0_recall, knn_b0_serendipity, 'o--', label=r'$\beta=0$')
    # tmp = zip(knn_b0_recall, knn_b0_serendipity)
    # for i, p in enumerate(tmp):
    #     ax[0].text(p[0], p[1], k_list[i], fontsize=11, ha='right',va='bottom')

    ax[0, 0].set_title('KNN popular')
    ax[0, 0].set_xlabel('Recall')
    ax[0, 0].set_ylabel(ser_label)
    ax[0, 0].grid(linewidth=0.5)

    ax[0, 1].plot(mf_recall, mf_serendipity, 'o-')
    tmp = zip(mf_recall, mf_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 1].text(p[0], p[1], components[i], fontsize=font_size, ha='right', va='bottom')
    ax[0, 1].set_title('Matrix factorization')
    ax[0, 1].set_xlabel('Recall')
    ax[0, 1].set_ylabel(ser_label)
    ax[0, 1].grid(linewidth=0.5)

    ax[1, 0].plot(autorec_recall, autorec_serendipity, 'o-', label='Serendipity')
    tmp = zip(autorec_recall, autorec_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 0].text(p[0], p[1], layers[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 0].set_title('AutoRec')
    ax[1, 0].set_xlabel('Recall')
    ax[1, 0].set_ylabel(ser_label)
    ax[1, 0].grid(linewidth=0.5)

    ax[1, 1].plot(ease_recall, ease_serendipity, 'o-', label='Serendipity')
    tmp = zip(ease_recall, ease_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 1].text(p[0], p[1], ease_lambdas[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 1].set_title('EASE')
    ax[1, 1].set_xlabel('Recall')
    ax[1, 1].set_ylabel(ser_label)
    ax[1, 1].grid(linewidth=0.5)

    plt.savefig(os.path.join(PATH, img_folder, 'recall_vs_serendipity' + path_postfix + img_ext))

    # Recall vs serendipity (fixed y)
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(16, 12))
    fig.suptitle('Recall vs serendipity (fixed axes)')

    plt.title('Recall versus serendipity (MovieLens-{})'.format(dataset_name))

    ax[0, 0].plot(knn_recall, knn_serendipity, 'o-')
    tmp = zip(knn_recall, knn_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 0].text(p[0], p[1], k_list[i], fontsize=font_size, ha='right', va='bottom')

    # ax[0].plot(knn_b0_recall, knn_b0_serendipity, 'o--', label=r'$\beta=0$')
    # tmp = zip(knn_b0_recall, knn_b0_serendipity)
    # for i, p in enumerate(tmp):
    #     ax[0].text(p[0], p[1], k_list[i], fontsize=11, ha='right',va='bottom')

    ax[0, 0].set_title('KNN popular')
    ax[0, 0].set_xlabel('Recall')
    ax[0, 0].set_ylabel(ser_label)
    ax[0, 0].set_xlim(0.99 * min_recall, 1.01 * max_recall)
    ax[0, 0].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[0, 0].grid(linewidth=0.5)

    ax[0, 1].plot(mf_recall, mf_serendipity, 'o-')
    tmp = zip(mf_recall, mf_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 1].text(p[0], p[1], components[i], fontsize=font_size, ha='right', va='bottom')
    ax[0, 1].set_title('Matrix factorization')
    ax[0, 1].set_xlabel('Recall')
    ax[0, 1].set_ylabel(ser_label)
    ax[0, 1].set_xlim(0.99 * min_recall, 1.01 * max_recall)
    ax[0, 1].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[0, 1].grid(linewidth=0.5)

    ax[1, 0].plot(autorec_recall, autorec_serendipity, 'o-', label='Serendipity')
    tmp = zip(autorec_recall, autorec_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 0].text(p[0], p[1], layers[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 0].set_title('AutoRec')
    ax[1, 0].set_xlabel('Recall')
    ax[1, 0].set_ylabel(ser_label)
    ax[1, 0].set_xlim(0.99 * min_recall, 1.01 * max_recall)
    ax[1, 0].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[1, 0].grid(linewidth=0.5)

    ax[1, 1].plot(ease_recall, ease_serendipity, 'o-', label='Serendipity')
    tmp = zip(ease_recall, ease_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 1].text(p[0], p[1], ease_lambdas[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 1].set_title('EASE')
    ax[1, 1].set_xlabel('Recall')
    ax[1, 1].set_ylabel(ser_label)
    ax[1, 1].set_xlim(0.99 * min_recall, 1.01 * max_recall)
    ax[1, 1].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[1, 1].grid(linewidth=0.5)

    plt.savefig(os.path.join(PATH, img_folder, 'recall_vs_serendipity_absolute' + path_postfix + img_ext))

    # Coverage vs serendipity
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(16, 12))

    fig.suptitle('Coverage versus serendipity')

    ax[0, 0].plot(knn_coverage, knn_serendipity, 'o-')
    tmp = zip(knn_coverage, knn_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 0].text(p[0], p[1], k_list[i], fontsize=font_size, ha='right', va='bottom')

    # ax[0].plot(knn_b0_coverage, knn_b0_serendipity, 'o--', label=r'$\beta=0$')
    # tmp = zip(knn_b0_coverage, knn_b0_serendipity)
    # for i, p in enumerate(tmp):
    #     ax[0].text(p[0], p[1], k_list[i], fontsize=11, ha='right',va='bottom')

    ax[0, 0].set_title('KNN popular')
    ax[0, 0].set_xlabel('Coverage')
    ax[0, 0].set_ylabel(ser_label)
    ax[0, 0].grid(linewidth=0.5)

    ax[0, 1].plot(mf_coverage, mf_serendipity, 'o-')
    tmp = zip(mf_coverage, mf_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 1].text(p[0], p[1], components[i], fontsize=font_size, ha='right', va='bottom')
    ax[0, 1].set_title('Matrix factorization')
    ax[0, 1].set_xlabel('Coverage')
    ax[0, 1].set_ylabel(ser_label)
    ax[0, 1].grid(linewidth=0.5)

    ax[1, 0].plot(autorec_coverage, autorec_serendipity, 'o-', label='Serendipity')
    tmp = zip(autorec_coverage, autorec_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 0].text(p[0], p[1], layers[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 0].set_title('AutoRec')
    ax[1, 0].set_xlabel('Coverage')
    ax[1, 0].set_ylabel(ser_label)
    ax[1, 0].grid(linewidth=0.5)

    ax[1, 1].plot(ease_coverage, ease_serendipity, 'o-', label='Serendipity')
    tmp = zip(ease_coverage, ease_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 1].text(p[0], p[1], ease_lambdas[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 1].set_title('EASE')
    ax[1, 1].set_xlabel('Coverage')
    ax[1, 1].set_ylabel(ser_label)
    ax[1, 1].grid(linewidth=0.5)

    plt.savefig(os.path.join(PATH, img_folder, 'coverage_vs_serendipity' + path_postfix + img_ext))

    # Coverage vs serendipity (fixed y)
    fig, ax = plt.subplots(2, 2, constrained_layout=True, figsize=(16, 12))

    fig.suptitle('Coverage versus serendipity (fixed axes)')

    ax[0, 0].plot(knn_coverage, knn_serendipity, 'o-')
    tmp = zip(knn_coverage, knn_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 0].text(p[0], p[1], k_list[i], fontsize=font_size, ha='right', va='bottom')

    # ax[0].plot(knn_b0_coverage, knn_b0_serendipity, 'o--', label=r'$\beta=0$')
    # tmp = zip(knn_b0_coverage, knn_b0_serendipity)
    # for i, p in enumerate(tmp):
    #     ax[0].text(p[0], p[1], k_list[i], fontsize=11, ha='right',va='bottom')

    ax[0, 0].set_title('KNN popular')
    ax[0, 0].set_xlabel('Coverage')
    ax[0, 0].set_ylabel(ser_label)
    ax[0, 0].set_xlim(0.99 * min_coverage, 1.01 * max_coverage)
    ax[0, 0].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[0, 0].grid(linewidth=0.5)

    ax[0, 1].plot(mf_coverage, mf_serendipity, 'o-')
    tmp = zip(mf_coverage, mf_serendipity)
    for i, p in enumerate(tmp):
        ax[0, 1].text(p[0], p[1], components[i], fontsize=font_size, ha='right', va='bottom')
    ax[0, 1].set_title('Matrix factorization')
    ax[0, 1].set_xlabel('Coverage')
    ax[0, 1].set_ylabel(ser_label)
    ax[0, 1].set_xlim(0.99 * min_coverage, 1.01 * max_coverage)
    ax[0, 1].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[0, 1].grid(linewidth=0.5)

    ax[1, 0].plot(autorec_coverage, autorec_serendipity, 'o-', label='Serendipity')
    tmp = zip(autorec_coverage, autorec_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 0].text(p[0], p[1], layers[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 0].set_title('AutoRec')
    ax[1, 0].set_xlabel('Coverage')
    ax[1, 0].set_ylabel(ser_label)
    ax[1, 0].set_xlim(0.99 * min_coverage, 1.01 * max_coverage)
    ax[1, 0].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[1, 0].grid(linewidth=0.5)

    ax[1, 1].plot(ease_coverage, ease_serendipity, 'o-', label='Serendipity')
    tmp = zip(ease_coverage, ease_serendipity)
    for i, p in enumerate(tmp):
        ax[1, 1].text(p[0], p[1], ease_lambdas[i], fontsize=font_size, ha='right', va='bottom')
    ax[1, 1].set_title('EASE')
    ax[1, 1].set_xlabel('Coverage')
    ax[1, 1].set_ylabel(ser_label)
    ax[1, 1].set_xlim(0.99 * min_coverage, 1.01 * max_coverage)
    ax[1, 1].set_ylim(0.99 * min_serendipity, 1.01 * max_serendipity)
    ax[1, 1].grid(linewidth=0.5)

    plt.savefig(os.path.join(PATH, img_folder, 'coverage_vs_serendipity_absolute' + path_postfix + img_ext))