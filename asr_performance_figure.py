import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 26})


def export_legend(legend, filename="legend.pdf"):
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)


def make_plot(measurements, labels, ax=None, width=0.4, y_lim=66, generalization=False, std=True, points=True):
    if ax is None:
        fig, ax = plt.subplots(layout='constrained', figsize=(8, 6))
        # fig, ax = plt.subplots(layout='constrained', figsize=(25, 20))
        plot = True
    else:
        plot = False
    groups = ("   Cross Entropy", "      Batch Hard", "SupCon")
    locations = (0.5 + 0.5 * width, 4 + 0.5 * width, 8 + 0.5 * width)

    x = np.arange(len(measurements) + 2)  # the label locations
    first_cut = 2
    second_cut = 5
    x = np.delete(x, first_cut)
    x = np.delete(x, second_cut)

    no_inters = [0, 2, 5]
    inters = [1, 3, 6]
    colors = ["lightblue", "orange", "lightgreen"]
    hatch = "//"
    alpha = 0.75

    for i, measurement in enumerate(measurements):
        if generalization:
            f1_idx = labels.index("F1@1(gen)")
            map_idx = labels.index("mAP@R(gen)")
        else:
            f1_idx = labels.index("F1@1")
            map_idx = labels.index("mAP@R(+)")

        run_data = np.array(measurement)
        f1 = run_data[:, f1_idx]
        f1_std = np.std(run_data[:, f1_idx])
        map = run_data[:, map_idx]
        map_std = np.std(run_data[:, map_idx])

        if i in no_inters:
            col = colors[0]
        elif i in inters:
            col = colors[1]
        else:
            col = colors[2]

        ax.bar(x[i], np.mean(f1), width, label=None, hatch=", ", edgecolor='black', color=col, zorder=2)
        ax.bar(x[i] + width, np.mean(map), width, label=None, alpha=0.75, hatch=hatch, edgecolor='black', color=col, zorder=2)

        if std:
            plt.errorbar(x[i], np.mean(f1), f1_std, fmt='.', color='Black', capsize=3)
            plt.errorbar(x[i] + width, np.mean(map), map_std, fmt='.', color='Black', capsize=3, zorder=2)

        if points:
            ax.scatter([x[i]] * len(f1), f1, c='k', s=10, zorder=2, alpha=alpha)
            ax.scatter([x[i] + width] * len(f1), map, c='k', s=10, zorder=2, alpha=alpha)

    legend_elements = [Patch(facecolor=colors[0], edgecolor='Black', label='Trained w/o interstates'),
                       Patch(facecolor="white", edgecolor='Black', label='F1@1'),
                       Patch(facecolor=colors[1], edgecolor='Black', label='Trained w interstates'),
                       Patch(facecolor="white", hatch=hatch, edgecolor='Black', label="MAP@R"),
                       Patch(facecolor=colors[2], edgecolor='Black', label='Trained w ISIL'),
                       ]
    # legend = ax.legend(handles=legend_elements, loc='upper left', ncols=3, frameon=False)
    # export_legend(legend)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_xticks(locations, groups)
    # ax.set_xticks(locations, groups, rotation=13)
    ax.set_ylim(0, y_lim)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    plt.axvline(x=2 + 0.5 * width, color='lightgray', linestyle='--', linewidth=1, zorder=1)

    if not plot:
        return ax


def make_error_plot(measurements, labels, ax=None, width=0.175, y_lim=100, std=True, points=True):
    if ax is None:
        fig, ax = plt.subplots(layout='constrained', figsize=(12, 4))
        # fig, ax = plt.subplots(layout='constrained', figsize=(40, 6))
        # y_lim = 150
        plot = True
    else:
        plot = False
    groups = ("Cross Entropy", "Batch Hard", "SupCon")
    locations = (3.75 * width, 4 - 0.5 * width, 7.5 - 0.5 * width)

    x = np.arange(len(measurements) + 2)  # the label locations
    first_cut = 2
    second_cut = 5
    x = np.delete(x, first_cut)
    x = np.delete(x, second_cut)

    no_inters = [0, 2, 5]
    inters = [1, 3, 6]
    colors = ["lightblue", "orange", "lightgreen"]
    hatch = "//"
    alpha = 0.5
    alpha_points = 0.75

    n_samples_c1 = 1144
    n_samples_c2 = 398
    n_samples_c3 = 85
    n_samples_c4 = 763
    n_samples = n_samples_c1 + n_samples_c2 + n_samples_c3 + n_samples_c4

    for i, measurement in enumerate(measurements):
        run_data = np.array(measurement)

        ap_c1 = run_data[:, labels.index("mAP cat I")]
        ap_c1_mean = np.mean(ap_c1)
        ap_c1_std = np.std(run_data[:, labels.index("mAP cat I")])
        ap_c2 = run_data[:, labels.index("mAP cat II")]
        ap_c2_mean = np.mean(ap_c2)
        ap_c2_std = np.std(run_data[:, labels.index("mAP cat II")])
        ap_c3 = run_data[:, labels.index("mAP cat III")]
        ap_c3_mean = np.mean(ap_c3)
        ap_c3_std = np.std(run_data[:, labels.index("mAP cat III")])
        ap_c4 = run_data[:, labels.index("mAP cat IV")]
        ap_c4_mean = np.mean(ap_c4)
        ap_c4_std = np.std(run_data[:, labels.index("mAP cat IV")])

        macro_avg = np.mean(run_data[:, labels.index("mAP cat I"):labels.index("mAP cat IV")+1], axis=1)  # [ap_c1_mean, ap_c2_mean, ap_c3_mean, ap_c4_mean]
        macro_avg_mean = np.mean(macro_avg)
        macro_std = np.std(np.mean(run_data[:, labels.index("mAP cat I"):labels.index("mAP cat IV")+1], axis=1))

        if i in no_inters:
            col = colors[0]
        elif i in inters:
            col = colors[1]
        else:
            col = colors[2]

        if first_cut <= i < second_cut:
            x_loc = x[i] - 0.5
        elif i >= second_cut:
            x_loc = x[i] - 1
        else:
            x_loc = x[i]

        ax.bar(x_loc, macro_avg_mean, width, label=None, hatch="", edgecolor='black', color=col, zorder=2, alpha=1)
        ax.bar(x_loc + 1 * width, ap_c1_mean, width, label=None, hatch="xx", edgecolor='black', color=col, zorder=2, alpha=alpha)
        ax.bar(x_loc + 2 * width, ap_c2_mean, width, label=None, hatch="//", edgecolor='black', color=col, zorder=2, alpha=alpha)
        ax.bar(x_loc + 3 * width, ap_c3_mean, width, label=None, hatch="--", edgecolor='black', color=col, zorder=2, alpha=alpha)
        ax.bar(x_loc + 4 * width, ap_c4_mean, width, label=None, hatch="..", edgecolor='black', color=col, zorder=2, alpha=alpha)

        if std:
            plt.errorbar(x_loc, macro_avg_mean, macro_std, fmt='.', color='Black', capsize=3, alpha=1)
            plt.errorbar(x_loc + 1 * width, ap_c1_mean, ap_c1_std, fmt='.', color='Black', capsize=3, alpha=alpha)
            plt.errorbar(x_loc + 2 * width, ap_c2_mean, ap_c2_std, fmt='.', color='Black', capsize=3, alpha=alpha)
            plt.errorbar(x_loc + 3 * width, ap_c3_mean, ap_c3_std, fmt='.', color='Black', capsize=3, alpha=alpha)
            plt.errorbar(x_loc + 4 * width, ap_c4_mean, ap_c4_std, fmt='.', color='Black', capsize=3, alpha=alpha)

        if points:
            ax.scatter([x_loc] * len(macro_avg), macro_avg, c='k', s=10, zorder=2, alpha=alpha_points)
            ax.scatter([x_loc + 1 * width] * len(ap_c1), ap_c1, c='k', s=10, zorder=2, alpha=alpha_points)
            ax.scatter([x_loc + 2 * width] * len(ap_c2), ap_c2, c='k', s=10, zorder=2, alpha=alpha_points)
            ax.scatter([x_loc + 3 * width] * len(ap_c3), ap_c3, c='k', s=10, zorder=2, alpha=alpha_points)
            ax.scatter([x_loc + 4 * width] * len(ap_c4), ap_c4, c='k', s=10, zorder=2, alpha=alpha_points)

            # ax.hlines(macro_avg, x_loc - width/10, x_loc + 3 * width + width/10, colors=['k'], lw=6, zorder=3)
        # ax.hlines(macro_avg, x_loc, x_loc + 3 * width, colors=['silver'], lw=3, zorder=3)
        # x_std = x_loc + 0.5 * (x_loc + 3 * width)
        # plt.errorbar(x_std, macro_avg, macro_std, fmt='.', color='Black', elinewidth=3, capsize=6, alpha=alpha)

    ax.set_ylim(50, y_lim)

    legend_elements = [
        Patch(facecolor="white", edgecolor='Black', label='I. Missing', hatch="xx"),
        Patch(facecolor="white", edgecolor='Black', label='II. Orientation', hatch="//"),
        Patch(facecolor="white", edgecolor='Black', label='III. Placement', hatch="--"),
        Patch(facecolor="white", edgecolor='Black', label='IV. Part-level', hatch=".."),
        Patch(facecolor="white", edgecolor='Black', label='Macro average', hatch=""),
        Patch(facecolor=colors[0], edgecolor='Black', label='Trained w/o interstates'),
        Patch(facecolor=colors[1], edgecolor='Black', label='Trained w interstates'),
        Patch(facecolor=colors[2], edgecolor='Black', label='Trained w ISIL'),

    ]
    # legend = ax.legend(handles=legend_elements, loc='upper left', ncols=8, frameon=False)
    # export_legend(legend)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average precision')
    ax.set_xticks(locations, groups)  # , rotation=13)
    ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=1)
    plt.axvline(x=2 + 0.25 * width, color='lightgray', linestyle='--', linewidth=1, zorder=1)

    if not plot:
        return ax


# plot options:
plot_points = False
plot_stds = True

columns = ["PR@1", "F1@1", "mAP@R(+)", "mAP@R(gen)", "F1@1(gen)", "mAP cat I", "mAP cat II", "mAP cat III", "mAP cat IV", "macro avg"]
resnet_ce = [
    [78.72, 38.72, 32.27, 10.76, 37.46, 95.81, 83.49, 43.30, 56.65],
    [78.41, 36.88, 31.08, 10.68, 39.10, 97.64, 85.99, 84.97, 73.71],
    [80.48, 41.13, 34.75, 10.19, 34.23, 98.55, 81.40, 72.33, 65.53],
    [80.82, 40.86, 25.49, 12.03, 40.10, 97.27, 88.09, 80.29, 69.52],
    [80.01, 40.04, 26.18, 12.30, 40.60, 96.79, 90.70, 67.83, 67.23],
]

resnet_ce_inters = [
    [82.61, 39.67, 34.54, 12.45, 41.77, 89.63, 83.04, 31.89, 57.92],
    [83.95, 40.79, 32.48, 11.77, 36.78, 95.56, 85.37, 73.84, 69.00],
    [82.32, 40.86, 33.80, 12.00, 39.86, 89.80, 77.46, 35.33, 54.52],
    [84.00, 43.56, 35.70, 13.68, 37.79, 96.27, 88.11, 76.09, 72.40],
    [84.44, 43.96, 36.53, 12.80, 43.88, 92.67, 80.58, 58.60, 72.14],
]

resnet_bh = [
    [78.58, 40.34, 40.35, 37.63, 60.39, 99.12, 76.10, 58.29, 57.09],
    [78.79, 44.43, 41.95, 36.27, 56.40, 98.46, 77.26, 70.63, 57.88],
    [79.68, 42.59, 43.16, 38.63, 60.17, 99.71, 78.68, 60.49, 57.44],
    [78.68, 43.51, 43.10, 38.84, 68.28, 98.76, 73.16, 84.95, 57.86],
    [79.21, 42.61, 43.36, 39.22, 60.77, 99.32, 83.93, 72.00, 66.11],
]

resnet_bh_inter = [
    [81.39, 37.02, 34.92, 31.56, 52.83, 97.37, 75.50, 52.10, 57.61],
    [81.97, 41.71, 38.32, 31.87, 54.76, 98.80, 87.17, 53.71, 65.37],
    [82.22, 39.27, 33.76, 32.74, 52.42, 97.52, 70.23, 40.54, 60.56],
    [83.37, 44.04, 41.52, 32.54, 52.05, 97.85, 86.35, 72.35, 75.27],
    [81.84, 42.58, 36.58, 29.82, 46.43, 97.71, 69.60, 37.69, 60.44],
]

resnet_bh_isil = [
    [82.61, 43.11, 45.87, 39.81, 57.58, 99.07, 68.31, 69.90, 58.59],
    [81.88, 41.67, 42.09, 38.05, 55.51, 98.04, 79.08, 42.36, 53.00],
    [82.73, 44.94, 43.42, 33.41, 55.60, 98.64, 76.15, 84.14, 68.78],
    [82.68, 47.75, 49.05, 34.18, 58.91, 97.47, 88.54, 62.10, 69.12],
    [82.12, 43.01, 44.80, 35.77, 55.53, 99.20, 79.02, 58.32, 71.28],
]

resnet_supcon = [
    [79.62, 40.59, 52.08, 34.26, 56.27, 99.63, 96.83, 69.60, 60.27],
    [79.64, 41.22, 49.89, 40.04, 58.37, 99.34, 90.64, 61.14, 62.95],
    [80.35, 42.20, 50.50, 39.98, 58.13, 99.67, 94.06, 68.96, 74.63],
    [79.61, 40.89, 47.58, 35.27, 57.18, 99.09, 95.28, 78.15, 63.51],
    [80.32, 37.95, 47.28, 38.58, 62.39, 99.39, 85.90, 71.50, 63.91],
]

resnet_supcon_inter = [
    [82.87, 47.33, 52.06, 32.82, 50.64, 99.64, 89.22, 74.50, 73.56],
    [82.39, 44.80, 54.50, 33.53, 51.54, 98.63, 93.59, 69.90, 66.14],
    [82.79, 48.20, 51.96, 36.47, 56.41, 99.34, 96.56, 60.76, 78.97],
    [82.52, 43.20, 50.90, 37.99, 59.61, 99.55, 94.35, 83.37, 79.90],
    [83.14, 47.39, 50.27, 33.53, 52.97, 98.55, 90.82, 58.83, 76.71],
]

resnet_supcon_isil = [
    [82.60, 46.53, 58.59, 35.41, 51.98, 99.92, 96.18, 73.83, 70.84],
    [82.59, 47.61, 57.81, 33.43, 54.16, 99.35, 89.68, 63.43, 69.42],
    [82.24, 41.72, 54.66, 36.90, 61.71, 99.88, 98.08, 77.28, 72.62],
    [83.52, 48.73, 61.77, 38.67, 67.55, 99.90, 97.07, 74.35, 80.91],
    [83.41, 49.60, 59.87, 34.90, 58.66, 99.53, 94.76, 63.99, 72.61],
]

resnet_performance = [resnet_ce, resnet_ce_inters, resnet_bh, resnet_bh_inter, resnet_bh_isil, resnet_supcon, resnet_supcon_inter, resnet_supcon_isil]
print(f"Showing performance on IndustReal test set")
make_plot(resnet_performance, columns, generalization=False, std=plot_stds, points=plot_points)
print(f"Showing: ResNet performance on IndustReal")

make_plot(resnet_performance, columns, generalization=True, std=plot_stds, points=plot_points)
print(f"Showing: ResNet performance on generalization")

make_error_plot(resnet_performance, columns, std=plot_stds, points=plot_points)
print(f"Showing: ResNet performance on error categories")
plt.show()

vit_ce = [
    [82.00, 47.90, 25.46, 13.03, 28.18, 98.44, 86.19, 81.75, 74.45],
    [83.10, 50.85, 29.58, 15.79, 32.90, 98.85, 75.77, 91.26, 68.04],
    [83.66, 49.44, 30.69, 12.13, 30.86, 98.28, 65.79, 88.30, 58.31],
    [83.15, 47.33, 28.53, 15.32, 38.73, 98.44, 81.93, 80.42, 69.35],
    [81.40, 42.45, 38.22, 10.22, 26.69, 89.85, 83.74, 61.80, 51.08],
]

vit_ce_inters = [
    [88.67, 54.22, 52.21, 15.13, 30.70, 97.53, 85.89, 72.25, 66.50],
    [88.84, 55.01, 54.04, 13.35, 31.09, 96.98, 84.84, 75.33, 60.01],
    [87.51, 53.02, 50.08, 11.89, 30.86, 96.53, 84.77, 83.88, 62.14],
    [87.99, 46.89, 47.27, 9.35, 33.11, 96.35, 83.88, 78.71, 63.98],
    [89.36, 53.94, 57.28, 14.45, 38.36, 98.20, 78.64, 78.79, 65.07],
]

vit_bh = [
    [78.73, 43.21, 50.21, 29.68, 46.70, 99.89, 96.83, 62.88, 54.15],
    [80.53, 47.48, 49.96, 29.85, 45.51, 99.92, 99.25, 66.79, 59.38],
    [79.47, 46.91, 51.42, 31.54, 43.05, 99.79, 92.02, 58.11, 64.08],
    [78.36, 42.85, 47.81, 33.93, 49.00, 99.32, 95.14, 74.25, 55.78],
    [79.29, 46.50, 49.98, 28.44, 41.65, 99.49, 89.81, 78.98, 57.97],
]

vit_bh_inter = [
    [84.50, 50.47, 54.76, 30.33, 44.97, 99.28, 89.73, 68.07, 67.93],
    [84.82, 53.68, 56.89, 28.43, 49.50, 98.99, 91.78, 69.48, 63.69],
    [84.60, 48.82, 53.00, 30.36, 43.17, 98.68, 91.58, 62.61, 63.08],
    [83.78, 42.75, 39.02, 29.50, 50.17, 97.89, 73.16, 82.59, 66.64],
    [84.40, 48.94, 54.65, 27.55, 45.18, 98.96, 90.44, 56.17, 66.77],
]

vit_bh_isil = [
    [84.77, 52.00, 57.60, 31.60, 46.22, 99.58, 95.51, 66.27, 69.51],
    [85.28, 50.91, 58.71, 30.24, 42.13, 99.68, 93.42, 54.40, 64.25],
    [85.11, 49.83, 58.22, 31.78, 44.07, 99.61, 96.42, 78.18, 65.52],
    [84.47, 49.38, 57.61, 30.12, 47.84, 99.58, 94.84, 67.93, 61.84],
    [84.78, 53.67, 56.76, 30.44, 48.14, 99.65, 90.94, 59.00, 62.65],
]

vit_supcon = [
    [80.18, 44.78, 51.95, 25.79, 41.29, 99.67, 75.45, 71.81, 61.38],
    [79.04, 45.81, 50.96, 24.65, 40.83, 99.90, 85.33, 65.77, 63.42],
    [79.27, 45.63, 52.05, 25.00, 43.36, 99.68, 96.50, 63.66, 60.85],
    [79.26, 45.19, 51.72, 31.46, 48.08, 99.39, 98.60, 91.96, 67.95],
    [78.86, 44.62, 50.55, 22.51, 38.62, 99.83, 77.13, 81.05, 62.76],
]

vit_supcon_inter = [
    [84.56, 51.44, 54.67, 26.91, 42.36, 99.17, 87.42, 89.33, 69.13],
    [84.16, 52.03, 62.52, 21.59, 41.73, 99.44, 92.64, 81.17, 71.45],
    [84.44, 52.45, 59.87, 24.57, 41.38, 99.24, 90.08, 73.75, 67.31],
    [84.26, 50.86, 59.11, 26.14, 42.39, 99.58, 96.59, 80.38, 61.45],
    [84.41, 53.47, 58.43, 23.87, 40.58, 99.35, 83.30, 60.14, 66.80],
]

vit_supcon_isil = [
    [84.48, 52.40, 58.72, 24.93, 42.71, 99.85, 88.38, 73.17, 70.94],
    [84.81, 55.93, 65.34, 27.42, 41.71, 99.47, 95.51, 68.75, 66.96],
    [84.76, 51.84, 60.25, 24.07, 39.06, 99.69, 97.57, 87.37, 75.55],
    [84.77, 52.32, 62.41, 26.85, 41.04, 99.78, 91.21, 73.07, 63.83],
    [84.57, 53.68, 62.10, 23.03, 40.37, 99.90, 93.13, 68.21, 69.76],
]

vit_performance = [vit_ce, vit_ce_inters, vit_bh, vit_bh_inter, vit_bh_isil, vit_supcon, vit_supcon_inter, vit_supcon_isil]
print(f"Showing performance on IndustReal test set")
make_plot(vit_performance, columns, generalization=False, std=plot_stds, points=plot_points)
print(f"Showing: vit performance on IndustReal")

make_plot(vit_performance, columns, generalization=True, std=plot_stds, points=plot_points)
print(f"Showing: vit performance on generalization")

make_error_plot(vit_performance, columns, std=plot_stds, points=plot_points)
print(f"Showing: vit performance on error categories")
plt.show()

