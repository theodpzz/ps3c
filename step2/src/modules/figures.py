import os
import seaborn as sns
import matplotlib.pyplot as plt

def plot_losses(args, logs):

    values = []
    epochs = list(logs.keys())

    for epoch in epochs:
        values.append(logs[epoch]["loss_train"])

    # plot
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=epochs, y=values, 
                 color='dodgerblue', linewidth=1.5, label='train', marker='o')
    plt.title('Losses VS Epochs');
    ax.set_yscale('log')

    # save figure
    path_figures = os.path.join(args.save_dir, "figures")
    path_figure  = os.path.join(path_figures, "losses.png")
    plt.savefig(path_figure)