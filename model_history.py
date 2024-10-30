import pickle


def get_model_history_save_path():
    """Returns the path where the model history is saved."""
    return "./model_evaluation/model_history.pkl"


def save_history(history: dict):
    """Saves the training history to a file."""
    history_dict = history.history
    save_path = get_model_history_save_path()
    with open(save_path, "wb") as f:
        pickle.dump(history_dict, f)


def load_history():
    """Loads and returns the saved training history."""
    save_path = get_model_history_save_path()
    with open(save_path, "rb") as f:
        history_dict = pickle.load(f)
    return history_dict
