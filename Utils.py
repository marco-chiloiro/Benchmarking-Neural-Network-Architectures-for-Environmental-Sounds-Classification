import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from keras.models import Model



def cross_validation(data, init_model, X_dim, n_folds=5, epochs=100, num_batch_size=32, verbose=1, transpose=False, init_AE=None, enc_layer=None, num_epochs_AE=None):
    """
    Perform cross-validation on the data.

    Parameters
    ----------
    data : DataFrame
        The data to split.
    init_model : function
        The function to initialize the model.
    n_folds : int
        The number of folds.
    epochs : int
        The number of epochs.
    num_batch_size : int
        The batch size.
    verbose : int
        The verbosity mode. (0 or 1)
    X_dim : tuple
        The dimensions of the input data (n_mel_bands, n_time_steps).
    transpose : bool
        Whether to transpose the input data. If True, the input data will be of shape (n_time_steps, n_mel_bands).
    init_AE : function
        The function to initialize the AE model.
    enc_layer : int
        The layer of the AE model to be used as encoder.
    num_epochs_AE : int
        The number of epochs for the AE model.

    Returns
    -------
    history : list
        The history of the training. (only classification model)
    """
    history = []
    time = []
    cm_list = []
    history_AE = None
    if init_AE is not None:
        history_AE = []
    for fold_k in range(1, n_folds+1):
        print(f'Fold {fold_k}/{n_folds}\n')
        if init_AE is not None:
            h, h_AE, t, cm = process_fold(init_model, fold_k, data, epochs, num_batch_size, verbose, X_dim, transpose, init_AE, enc_layer, num_epochs_AE)
            history_AE.append(h_AE)
        else:
            h, t, cm = process_fold(init_model, fold_k, data, epochs, num_batch_size, verbose, X_dim, transpose, init_AE, enc_layer)
        history.append(h)
        time.append(t)
        cm_list.append(cm)
        # plot results
        print('Results for fold', fold_k, ':\n')
        if init_AE is not None:
            print('AE training results:')
            show_results_AE(history_AE[-1])
        print('Classification training results:')
        show_results(history[-1], cm_list[-1])

    # average results
    print('Average results:\n')    
    print_average_results(history, cm_list, n_folds, time, history_AE)

    # plot average results
    print('Plotting average results:\n')
    plot_average_results(history, n_folds, epochs, history_AE, num_epochs_AE)
                    
    # plot average confusion matrix
    print('Average confusion matrix:\n')
    avg_cm = np.mean(cm_list, axis=0)
    std_cm = np.std(cm_list, axis=0)
    plot_confusion_matrix(avg_cm, std_cm)
    
    return history



def process_fold(init_model, fold_k, data, epochs, num_batch_size, verbose, X_dim, transpose, init_AE=None, enc_layer=None, num_epochs_AE=None):
    """
    Process a fold of the cross-validation. It uses the fold_k for testing and the rest for training.

    Parameters
    ----------
    init_model : function
        The function to initialize the model.
    fold_k : int
        The fold to be used for testing.
    data : DataFrame
        The data to split.
    epochs : int
        The number of epochs.
    num_batch_size : int
        The batch size.
    verbose : int
        The verbosity mode. (0 or 1)
    X_dim : tuple
        The dimensions of the input data (n_mel_bands, n_time_steps).
    transpose : bool
        Whether to transpose the input data. If True, the input data will be of shape (n_time_steps, n_mel_bands).
    init_AE : function
        The function to initialize the AE model.
    enc_layer : int
        The layer of the AE model to be used as encoder.
    num_epochs_AE : int
        The number of epochs for the AE model.

    Returns
    -------
    history : list
        The history of the training. (only classification model)
    time : float
        The time taken to train the model.
    cm : ndarray    
        The confusion matrix.
    history_AE : list
        The history of the training. (only if AE model is used)
    """
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(fold_k, data, X_dim, transpose)

    if init_AE is not None:
        # initialize the AE model
        AE = init_AE()
        print('Training AE model...\n')
        # train the AE model
        start = datetime.now()
        history_AE = AE.fit(X_train, X_train, batch_size=num_batch_size, epochs=num_epochs_AE, validation_data=(X_test, X_test), verbose=verbose)
        end = datetime.now()
        time_AE = end - start
        print("Training AE completed in time: ", time_AE, "\n")
        # save and freeze the encoder part
        encoder = Model(inputs=AE.input, outputs=AE.layers[enc_layer].output)
        encoder.trainable = False
        # initialize the model
        model = init_model(encoder)
        print('Training classification model...\n')
        # pre-training accuracy
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Pre-training accuracy: {100*score[1]} %')
        # train the model
        start = datetime.now()
        history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=verbose)
        end = datetime.now()
        time_classification = end - start
        print("Training completed in time: ", time_classification, "\n")
        # confusion matrix
        cm = compute_confusion_matrix(model, X_test, y_test)
        time = time_AE.total_seconds() + time_classification.total_seconds()
        return history, history_AE, time, cm

    else:
        # initialize the model
        model = init_model()
        # pre-training accuracy
        score = model.evaluate(X_test, y_test, verbose=0)
        print(f'Pre-training accuracy: {100*score[1]} %')
        # train the model
        start = datetime.now()
        history = model.fit(X_train, y_train, batch_size=num_batch_size, epochs=epochs, validation_data=(X_test, y_test), verbose=verbose)
        end = datetime.now()
        time = end - start
        print("Training completed in time: ", time), 
        # confusion matrix
        cm = compute_confusion_matrix(model, X_test, y_test)
        return history, time.total_seconds(), cm



def train_test_split(fold_k, data, X_dim, transpose):
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    fold_k : int
        The fold to be used for testing.
    data : DataFrame
        The data to be split.
    X_dim : tuple
        The dimensions of the input data. MUST be (n_mel_bands, n_time_steps).
    transpose : bool
        Whether to transpose the input data. If True, the input data will be of shape (n_time_steps, n_mel_bands).

    Returns
    -------
    XX_train : ndarray
        The training data.
    XX_test : ndarray
        The testing data.
    yy_train : ndarray
        The training labels.
    yy_test : ndarray
        The testing labels.
    """
    # keep all date except fold_k (even augmented data)
    X_train = np.stack(data[data.fold != fold_k].mel_spectrogram.to_numpy())
    # keep only original data for testing
    X_test = np.stack(data[(data.fold == fold_k) & (data.original == True)].mel_spectrogram.to_numpy())

    y_train = data[data.fold != fold_k].target.to_numpy()
    y_test = data[(data.fold == fold_k) & (data.original == True)].target.to_numpy()

    XX_train = X_train.reshape(X_train.shape[0], *X_dim)
    XX_test = X_test.reshape(X_test.shape[0], *X_dim)

    # Converts a class vector (integers) to binary class matrix.
    yy_train = to_categorical(y_train)
    yy_test = to_categorical(y_test)

    if transpose:
        XX_train = np.array([np.transpose(x) for x in X_train])
        XX_test = np.array([np.transpose(x) for x in X_test])
    
    return XX_train, XX_test, yy_train, yy_test



def compute_confusion_matrix(model, X_test, y_test):
    """
    Compute the confusion matrix given the model and the testing data.

    Parameters
    ----------
    model : Model
        The model.
    X_test : ndarray
        The testing data.
    y_test : ndarray
        The testing labels.

    Returns
    -------
    cm : ndarray
        The confusion matrix.
    """
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    return cm



def print_average_results(history, cm_list, n_folds, time, history_AE=None):
    """
    Print the average results of the training.

    Parameters
    ----------
    history : list
        The history of the training.
    cm_list : list
        The confusion matrices.
    n_folds : int
        The number of folds.
    time : list
        The time taken to train the model for each fold.
    history_AE : list
        The history of the training. (only if AE model is used)
    """
    # averages
    avg_acc = np.mean([history[i].history['accuracy'][-1] for i in range(n_folds)])
    avg_val_acc = np.mean([history[i].history['val_accuracy'][-1] for i in range(n_folds)])
    avg_loss = np.mean([history[i].history['loss'][-1] for i in range(n_folds)])
    avg_val_loss = np.mean([history[i].history['val_loss'][-1] for i in range(n_folds)])
    avg_time = np.mean(time)
    if history_AE is not None:
        avg_loss_AE = np.mean([history_AE[i].history['loss'][-1] for i in range(n_folds)])
        avg_val_loss_AE = np.mean([history_AE[i].history['val_loss'][-1] for i in range(n_folds)])
    # stds
    std_acc = np.std([history[i].history['accuracy'][-1] for i in range(n_folds)])
    std_val_acc = np.std([history[i].history['val_accuracy'][-1] for i in range(n_folds)])
    std_loss = np.std([history[i].history['loss'][-1] for i in range(n_folds)])
    std_val_loss = np.std([history[i].history['val_loss'][-1] for i in range(n_folds)])
    std_time = np.std(time)
    if history_AE is not None:
        std_loss_AE = np.std([history_AE[i].history['loss'][-1] for i in range(n_folds)])
        std_val_loss_AE = np.std([history_AE[i].history['val_loss'][-1] for i in range(n_folds)])

    # print results
    if history_AE is not None:
        print('AE training results:')
        print(f'Average training loss: {"{:.3f}".format(avg_loss_AE)} ± {"{:.3f}".format(std_loss_AE)}')
        print(f'Average validation loss: {"{:.3f}".format(avg_val_loss_AE)} ± {"{:.3f}".format(std_val_loss_AE)}\n')
    print('Classification training results:')
    print(f'Average training accuracy: {"{:.2f}".format(avg_acc)} ± {"{:.2f}".format(std_acc)}')
    print(f'Average validation accuracy: {"{:.2f}".format(avg_val_acc)} ± {"{:.2f}".format(std_val_acc)}')
    print(f'Average training loss: {"{:.2f}".format(avg_loss)} ± {"{:.2f}".format(std_loss)}')
    print(f'Average validation loss: {"{:.2f}".format(avg_val_loss)} ± {"{:.2f}".format(std_val_loss)}\n')

    print(f'Average (total) training time: {"{:.2f}".format(avg_time)} ± {"{:.2f}".format(std_time)} s\n')



def plot_average_results(history, n_folds, epochs, history_AE=None, num_epochs_AE=None):
    """
    Plot the average results of the training.
    
    Parameters
    ----------
    history : list
        The history of the training (classification model).
    n_folds : int
        The number of folds.
    epochs : int
        The number of epochs.
    history_AE : list
        The history of the training (AE model).
    num_epochs_AE : int
        The number of epochs for the AE model.
    """
    # plot average results
    mean_acc = [np.mean([history[i].history['accuracy'][j] for i in range(n_folds)]) for j in range(epochs)]
    mean_val_acc = [np.mean([history[i].history['val_accuracy'][j] for i in range(n_folds)]) for j in range(epochs)]
    mean_loss = [np.mean([history[i].history['loss'][j] for i in range(n_folds)]) for j in range(epochs)]
    mean_val_loss = [np.mean([history[i].history['val_loss'][j] for i in range(n_folds)]) for j in range(epochs)]
    std_acc = [np.std([history[i].history['accuracy'][j] for i in range(n_folds)]) for j in range(epochs)]
    std_val_acc = [np.std([history[i].history['val_accuracy'][j] for i in range(n_folds)]) for j in range(epochs)]
    std_loss = [np.std([history[i].history['loss'][j] for i in range(n_folds)]) for j in range(epochs)]
    std_val_loss = [np.std([history[i].history['val_loss'][j] for i in range(n_folds)]) for j in range(epochs)]
    if history_AE is not None:
        mean_loss_AE = [np.mean([history_AE[i].history['loss'][j] for i in range(n_folds)]) for j in range(num_epochs_AE)]
        mean_val_loss_AE = [np.mean([history_AE[i].history['val_loss'][j] for i in range(n_folds)]) for j in range(num_epochs_AE)]
        std_loss_AE = [np.std([history_AE[i].history['loss'][j] for i in range(n_folds)]) for j in range(num_epochs_AE)]
        std_val_loss_AE = [np.std([history_AE[i].history['val_loss'][j] for i in range(n_folds)]) for j in range(num_epochs_AE)]

        print('AE training result plots:')
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Mean Loss')
        axs.plot(range(num_epochs_AE), mean_loss_AE, color='tab:red', linewidth=1, label='Train')
        axs.fill_between(range(num_epochs_AE), np.array(mean_loss_AE) - np.array(std_loss_AE), np.array(mean_loss_AE) + np.array(std_loss_AE), color='tab:red', alpha=0.3)
        axs.plot(range(num_epochs_AE), mean_val_loss_AE, color='tab:blue', linewidth=1, label='Validation')
        axs.fill_between(range(num_epochs_AE), np.array(mean_val_loss_AE) - np.array(std_val_loss_AE), np.array(mean_val_loss_AE) + np.array(std_val_loss_AE), color='tab:blue', alpha=0.3)
        axs.legend(loc='upper right')
        axs.grid(True)
        plt.show()

    print('Classification training result plots:')
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Mean Accuracy')
    axs[0].plot(range(epochs), mean_acc, color='tab:red', linewidth=1, label='Train')
    axs[0].fill_between(range(epochs), np.array(mean_acc) - np.array(std_acc), np.array(mean_acc) + np.array(std_acc), color='tab:red', alpha=0.3)
    axs[0].plot(range(epochs), mean_val_acc, color='tab:blue', linewidth=1, label='Validation')
    axs[0].fill_between(range(epochs), np.array(mean_val_acc) - np.array(std_val_acc), np.array(mean_val_acc) + np.array(std_val_acc), color='tab:blue', alpha=0.3)
    axs[0].legend(loc='upper left')
    axs[0].grid(True)

    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Mean Loss')
    axs[1].plot(range(epochs), mean_loss, color='tab:red', linewidth=1, label='Train')
    axs[1].fill_between(range(epochs), np.array(mean_loss) - np.array(std_loss), np.array(mean_loss) + np.array(std_loss), color='tab:red', alpha=0.3)
    axs[1].plot(range(epochs), mean_val_loss, color='tab:blue', linewidth=1, label='Validation')
    axs[1].fill_between(range(epochs), np.array(mean_val_loss) - np.array(std_val_loss), np.array(mean_val_loss) + np.array(std_val_loss), color='tab:blue', alpha=0.3)
    axs[1].legend(loc='upper right')
    axs[1].grid(True)
    plt.show()



def plot_confusion_matrix(cm, std=None):
    """
    Plot the confusion matrix.
    
    Parameters
    ----------
    cm : ndarray
        The confusion matrix.
    std : ndarray
        The standard deviation of the confusion matrix.
    """
    # percentage
    cm = np.round(cm*100, 2)
    if std is not None:
        std = np.round(std*100, 2)
    # Plot confusion matrix with standard deviations
    fig, ax = plt.subplots(figsize=(13, 13))
    label_names = ['dog', 'rain', 'sea_waves', 'crying_baby', 'clock_tick', 'sneezing', 'helicopter', 'chainsaw', 'rooster', 'crackling_fire']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, values_format='g', cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    if std is not None:
    # Annotate standard deviations below the mean values
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if std[i, j] > 0:
                    std_val = std[i, j]
                    ax.text(j, i + 0.25, f"±{std_val:.2f}", ha='center', va='center', fontsize=10, color='red')

    plt.title('Confusion matrix (%)')
    plt.show()



def show_results(history, cm=None):
    """
    Show the results of the training.

    Parameters
    ----------
    history : list
        The history of the training.
    cm : ndarray
        The confusion matrix.
    """
    # Show accuracy and loss for training and validation sets, in two different plots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_ylabel('Accuracy')
    axs[0].grid(True)

    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_ylabel('Loss')
    axs[1].grid(True)

    for ax in axs:
        ax.set_xlabel('Epoch')
        ax.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    if cm is not None:
        plot_confusion_matrix(cm)



def show_results_AE(history):
    """
    Show the results of the training of the AE model. (then only the loss is plotted)

    Parameters
    ----------
    history : list
        The history of the training.
    """
    # plot only the loss
    fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    axs.set_xlabel('Epoch')
    axs.set_ylabel('Loss')
    axs.legend(['Train', 'Validation'], loc='upper left')
    axs.grid(True)
    plt.show()



def create_shifted_data(X):
    """
    Create shifted data. Shifts the data by one time step and fills the first time step with zeros.

    Parameters
    ----------
    X : ndarray
        The data to be shifted.

    Returns
    -------
    X_shifted : ndarray
        The shifted data.
    """
    X_inv = X[:, ::-1, :]
    X_shifted = np.zeros(X_inv.shape)
    X_shifted[:, 1:, :] = X_inv[:, :-1, :]
    return X_shifted
