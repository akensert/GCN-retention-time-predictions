import numpy as np


def mean_squared_error(y_true, y_pred):
    y_pred = np.reshape(y_pred, (y_true.shape))
    return np.mean(np.square(y_true - y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    y_pred = np.reshape(y_pred, (y_true.shape))
    return np.mean(np.abs(y_true - y_pred))

def mean_absolute_percentage_error(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred) * 100

def mean_relative_error(y_true, y_pred):
    y_pred = np.reshape(y_pred, (y_true.shape))
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1))

def mean_relative_percentage_error(y_true, y_pred):
    return mean_relative_error(y_true, y_pred) * 100

dispatcher = {
    'mse': mean_squared_error,
    'mean_squared_error': mean_squared_error,
    'rmse': root_mean_squared_error,
    'root_mean_squared_error': root_mean_squared_error,
    'mae': mean_absolute_error,
    'mean_absolute_error': mean_absolute_error,
    'mape': mean_absolute_percentage_error,
    'mean_absolute_percentage_error': mean_absolute_percentage_error,
    'mre': mean_relative_error,
    'mean_relative_error': mean_relative_error,
    'mrpe': mean_relative_percentage_error,
    'mean_relative_percentage_error': mean_relative_percentage_error,
}

def get(identifier):
    if identifier is None:
        return dispatcher['mae']
    elif identifier.lower() in dispatcher.keys():
        return dispatcher[identifier.lower()]
    else:
        raise TypeError(
            'Could not interpret metric identifier; has to be one of: "' +
            '", "'.join([k for k in dispatcher.keys()]) + '".'
        )
