import matplotlib.pyplot as plt


def create_loading():
    """Create the loading bar by analysing the plot"""
    plt.ion()
    fig = plt.figure(num=None, figsize=(7, 2), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Loading')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2)
    name = ["Loading"]
    ax.barh(name, [0], align='center', color='orange')
    ax.set_xlim([0, 100])
    ax.set_yticks(name)
    ax.set_yticklabels(name)
    plt.draw()
    return ax, name, fig


def update_loading(ax, name, percent):
    """Update the percent of the progress bar"""
    actual = [percent]
    ax.clear()
    ax.barh(name, actual, align='center', color='orange')
    ax.set_xlim([0, 100])
    ax.set_yticks(name)
    ax.set_yticklabels(name)
    plt.draw()
    try:
        plt.pause(0.1)  # is necessary for the plot to update for some reason
    except:
        return False
    return True


def close_loading(fig):
    """Just close the progress bar which is a figure"""
    plt.close(fig)
