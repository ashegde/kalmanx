import matplotlib.pyplot as plt
import mlx.core as mx

def plot_data(data: mx.array,
              ax_titles: list[str],
              labels: list[str],
              title: str,
              figsize: tuple,
              colors: list[str],
              linestyles: list[str],
              markers: list[str],
              save_path: str) -> None:
    
    # hardwired for 
    fig, axs = plt.subplots(3, 2, figsize=figsize, sharex=True)

    for k in range(data.shape[0]):  # Over datasets/runs
        for i in range(3):  # Over dimensions x, y, z
            # Position subplot (left column)
            axs[i, 0].plot(
                data[k, :, i],
                color=colors[k],
                linestyle=linestyles[k],
                marker=markers[k],
                label=labels[k],
                alpha=0.4,
            )
            axs[i, 0].set_title(ax_titles[i], fontsize=10)
            axs[i, 0].set_ylabel('Value')
            axs[i, 0].grid(True)
            if i == 2:
                axs[i, 0].set_xlabel('Iteration')

            # Velocity subplot (right column)
            axs[i, 1].plot(
                data[k, :, i + 3],
                color=colors[k],
                linestyle=linestyles[k],
                marker=markers[k],
                label=labels[k],
                alpha=0.4,
            )
            axs[i, 1].set_title(ax_titles[i + 3], fontsize=10)
            axs[i, 1].set_ylabel('Value')
            axs[i, 1].grid(True)
            if i == 2:
                axs[i, 1].set_xlabel('Iteration')
    
        handles0, labels0 = axs[0, 0].get_legend_handles_labels()
        axs[0, 0].legend(handles0, labels0)


    plt.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust for suptitle space
    plt.savefig(save_path)
    plt.close()