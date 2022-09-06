import matplotlib.pyplot as plt


def plot_hist(hist, filename=None):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title(f"model accuracy \n {filename}")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.show()
    plt.savefig(filename)
