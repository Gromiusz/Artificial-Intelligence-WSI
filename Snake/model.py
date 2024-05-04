from enum import Enum
import pickle
import numpy as np
# from sklearn.linear_model import LogisticRegression
from classifier import MyLogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache
from data import consolidate_data, delete_disturbing_data
import matplotlib.pyplot as plt

"""Implement your model, training code and other utilities here. Please note, you can generate multiple 
pickled data files and merge them into a single data list."""


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

# zapis modelu, aby nie musial byc uczony za kazdym razem
@lru_cache(maxsize=None)
def learn_model_cached():
    return learn_model()


def game_state_to_data_sample(game_state: dict):
    raise NotImplementedError()


def get_predicted_move(game_state, bounds=(300, 300), model=None):
    if model is None:
        model = learn_model_cached()

    attributes = np.array(generate_attributes_for_state(game_state, bounds))
    attributes = attributes.reshape(1, -1)
    predicted_move = model.predict(attributes)
    return predicted_move


def generate_attributes(game_states, bounds):
    print("Generating attributes...", end = ' ')
    attributes_list = []
    labels_list = []
    for state in game_states:
        attributes = generate_attributes_for_state(state[0], bounds)
        attributes_list.append(attributes)
        labels_list.append(state[1])
    print("Done")
    return attributes_list, labels_list


def generate_attributes_for_state(state, bounds):
    # print(state)
    foodPosition = state['food']
    snakeBody = state['snake_body']
    headPosition = snakeBody[-1]
    actualDirection = state['snake_direction']
    
    attributes = []

    for direction in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
        neighbor_position = get_neighbor_position(headPosition, direction)

        if is_obstacle(neighbor_position, snakeBody, bounds):
            attributes.append(1)
        else:
            attributes.append(0)

        if is_food_in_direction(headPosition, direction, foodPosition):
            attributes.append(1)
        else:
            attributes.append(0)

        if is_this_direction_actual(direction.value, actualDirection.value):
            attributes.append(1)
        else:
            attributes.append(0)
        
    return attributes


def get_neighbor_position(head_position, direction):
    x, y = head_position
    if direction == Direction.UP:
        return x, y - 30
    elif direction == Direction.RIGHT:
        return x + 30, y
    elif direction == Direction.DOWN:
        return x, y + 30
    elif direction == Direction.LEFT:
        return x - 30, y


def is_food_in_direction(head_position, direction, food_position):
    x, y = head_position
    xf, yf = food_position

    if direction == Direction.UP and y <= yf:
        return True
    elif direction == Direction.RIGHT and x <= xf:
        return True
    elif direction == Direction.DOWN and y >= yf:
        return True
    elif direction == Direction.LEFT and x >= xf:
        return True
    else:
        return False
    

def is_this_direction_actual(direction, actualDirection):
    if direction == actualDirection:
        return True
    else:
        return False


def is_obstacle(position, snake_body, bounds):
    x, y = position
    if x < 0 or x >= bounds[0] or y < 0 or y >= bounds[1]:
        return True
    if position in snake_body:
        return True
    return False


def learn_model():
    merged_data, bounds = consolidate_data(
        "data/2024-04-12_15-35-48-dobre.pickle",
        "data/2024-04-12_20-57-44-n.pickle",
        "data/2024-04-12_21-12-06-n.pickle",
        "data/2024-04-14_15-28-57-n.pickle"
    )
    data = delete_disturbing_data(merged_data, bounds)
    attributes, labels = generate_attributes(data, bounds)
    # print(attributes)
    print("Learning model...", end = ' ')
    attributes_ndarray = np.array(attributes)
    labels_ndarray = np.array(labels)
    X = attributes_ndarray  # Dane wejściowe
    y = labels_ndarray  # Etykiety akcji w grze
    y_str = [direction.value for direction in y]

    # Przekształcenie etykiet
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = MyLogisticRegression(n_iters=1000)
    model.fit(X_train, y_train)
    print("Done")
    return model


if __name__ == "__main__":
    merged_data, bounds = consolidate_data(
        "data/2024-04-12_15-35-48-dobre.pickle",
        "data/2024-04-12_20-57-44-n.pickle",
        "data/2024-04-12_21-12-06-n.pickle",
        "data/2024-04-14_15-28-57-n.pickle"
    )
    data = delete_disturbing_data(merged_data, bounds)

    attributes, labels = generate_attributes(data, bounds)
    # print(attributes)
    # attributes_ndarray = np.array(attributes)
    # labels_ndarray = np.array(labels)

    # X = attributes_ndarray  # Dane wejściowe
    # y = labels_ndarray  # Etykiety
    X = np.array(attributes)
    lenx = len(X)
    sh = X.shape[0]
    y = np.array(labels)
    y_str = [direction.value for direction in y]


    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)

    for train_size, test_size in [(0.008, 0.002), (0.08, 0.02), (0.8, 0.2)]:
    # for train_size, test_size in [(0.8, 0.2)]:

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, train_size=train_size, test_size=test_size, random_state=42)
        # Trenowanie modelu regresji logistycznej
        model = MyLogisticRegression(n_iters=1000)
        model.fit(X_train, y_train)

        # Ocena modelu
        y_pred_train = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        y_pred_test = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        print('\n')
        print(f"Train size: {train_size*100}%, Test size: {test_size*100}%, so {train_size*100+test_size*100}% of the data")
        print(f"Accuracy on the training set: {accuracy_train}")
        print(f"Accuracy on the test set: {accuracy_test}")

 
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    # Lista wartości parametru C do przetestowania
    C_values = [0.001, 0.01, 0.1, 0.5, 1, 2, 10]
    accuracy_train_list = []
    accuracy_test_list = []
    print("\n")
    # Przebadanie różnych wartości parametru C
    for C in C_values:
        model = MyLogisticRegression(C=C, n_iters=1000)
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        accuracy_train = accuracy_score(y_train, y_pred_train)
        y_pred_test = model.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        print(f"Parameter C: {C}")
        print(f"Accuracy on the training set: {accuracy_train}")
        print(f"Accuracy on the test set: {accuracy_test}")
        accuracy_train_list.append(accuracy_train)
        accuracy_test_list.append(accuracy_test)
        print("\n")

    plt.plot(C_values, accuracy_train_list, label = "Training accuracy")
    plt.plot(C_values, accuracy_test_list, label = "Test accuracy")
    plt.xscale('log')
    plt.xlabel('Parameter C (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy in relation to parameter C')
    plt.legend()
    plt.show()

