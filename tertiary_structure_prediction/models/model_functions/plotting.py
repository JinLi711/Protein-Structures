import matplotlib.pyplot as plt

def plot_loss (history):
    """
    Plot loss against epochs
    
    :param history: The history of training for model
    :type  history: keras.callbacks.History
    """
    
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot (epochs, loss_values, 'bo', label='Training loss')
    plt.plot (epochs, val_loss_values, 'b', label="validation loss")
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_acc (history, acc='acc', val_acc='val_acc'):
    """
    Plot accuracy against epochs
    
    :param history: The history of training for model
    :type  history: keras.callbacks.History
    :param acc: Name of train accuracy key
    :type  acc: str
    :param val_acc: Name of validation accuracy key
    :type  val_acc: str
    """
    
    history_dict = history.history
    acc = history_dict[acc]
    val_acc = history_dict[val_acc]
    loss_values = history_dict['loss']
    epochs = range(1, len(loss_values) + 1)

    plt.plot (epochs, acc, 'bo', label='Training accuracy')
    plt.plot (epochs, val_acc, 'b', label="validation accuracy")
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def smooth_curve(points, factor=0.8):
    """
    Smooth out a list of points

    :param points: list of y coordinate points
    :type  points: list
    :param factor: smoothness factor
    :type  factor: float
    :return: Smoothed out elements of a list
    :rtype:  factor: list
    """

    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_smooth(history, acc='acc', val_acc='val_acc'):
    """
    Plot accuracy and loss graph, but with exponentially smoothed.
    
    :param history: The history of training for model
    :type  history: keras.callbacks.History
    :param acc: Name of train accuracy key
    :type  acc: str
    :param val_acc: Name of validation accuracy key
    :type  val_acc: str
    """

    acc = history.history[acc]
    val_acc = history.history[val_acc]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs,
             smooth_curve(acc), 'bo', label='Smoothed training acc')
    plt.plot(epochs,
             smooth_curve(val_acc), 'b', label='Smoothed validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs,
             smooth_curve(loss), 'bo', label='Smoothed training loss')
    plt.plot(epochs,
             smooth_curve(val_loss), 'b', label='Smoothed validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()