
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#=======================================================================================================================
def plot_histogram( x, dir, mode ):

    fig = plt.figure()

    # the histogram of the data
    n, bins, patches = plt.hist(x, 100, normed=True, facecolor='blue', alpha=0.5)

    plt.xlabel('Log-likelihood value')
    plt.ylabel('Probability')
    plt.grid(True)

    plt.savefig(dir + 'histogram_' + mode + '.png', bbox_inches='tight')
    plt.close(fig)

#=======================================================================================================================
def plot_images( args, x_sample, dir, file_name, size_x=3, size_y=3):

    fig = plt.figure(figsize=(size_x, size_y))
    # fig = plt.figure(1)
    gs = gridspec.GridSpec(size_x, size_y)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(x_sample):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape((args.input_size[1], args.input_size[2])), cmap='Greys_r')

    plt.savefig(dir + file_name + '.png', bbox_inches='tight')
    plt.close(fig)