import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.ion()
    fig = plt.figure(num=None, figsize=(7, 2), dpi=80, facecolor='w', edgecolor='k')
    fig.canvas.set_window_title('Loading')
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.2)
    name = ["Loading"]
    actual = [0]
    ax.barh(name, actual, align='center', color='orange')
    ax.set_xlim([0, 100])
    ax.set_yticks(name)
    ax.set_yticklabels(name)
    plt.draw()
    for i in range(120):
        actual = [i]
        ax.clear()
        ax.barh(name, actual, align='center', color='orange')
        ax.set_xlim([0, 100])
        ax.set_yticks(name)
        ax.set_yticklabels(name)
        plt.draw()
        plt.pause(0.1)  # is necessary for the plot to update for some reason
