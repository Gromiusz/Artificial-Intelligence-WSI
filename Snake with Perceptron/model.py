from enum import Enum
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from functools import lru_cache
from data import consolidate_data, delete_disturbing_data
import matplotlib.pyplot as plt
import torch
from BCDataset import BCDataset
from MLP import MLP
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
import torch.optim as optim

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
    attributes = torch.tensor(attributes).float()  # Konwersja na typ Float
    # Wczytanie stanu modelu
    state_dict = torch.load('best_model.pth')
    
    # Wyodrębnienie modelu z stanu
    input_size = len(attributes[0])
    hidden_size = 64
    output_size = 4
    new_model = MLP(input_size, hidden_size, output_size)  # Utwórz nową instancję modelu MLP
    new_model.load_state_dict(state_dict)  # Załaduj stan modelu
    new_model.eval()  # Ustaw model w trybie ewaluacji

    with torch.no_grad():  # Wyłącz obliczanie gradientów
        probabilities = new_model(attributes)
    probabilities = probabilities[0]  # Pobranie prawdopodobieństw klasy
    max_probability_index = torch.argmax(probabilities)
    predicted_move = max_probability_index.item()  # Pobranie przewidywanej klasy jako wartość int

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
    attributes_ndarray = np.array(attributes)
    labels_ndarray = np.array(labels)

    X = attributes_ndarray.tolist()  # Dane wejściowe
    y = labels_ndarray.tolist()  # Etykiety
    y = [direction.value for direction in y]

    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42)

    input_size = len(X_train[0])  # Rozmiar wejścia to liczba cech w jednej próbce danych
    hidden_size = 64
    output_size = 4  
    num_epochs = 20
    learning_rate = 0.001
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'best_model.pth'

    train_dataset = BCDataset(X_train, y_train)
    val_dataset = BCDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size, hidden_size, output_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)

    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path):
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Ustawienie modelu w trybie trenowania
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for batch in train_loader:
            inputs, labels = batch['data'].to(device), batch['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Ustawienie modelu w trybie ewaluacji
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch['data'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
        
        # Obliczanie dokładności na zbiorze walidacyjnym
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Zapis modelu, jeśli uzyskano lepszą dokładność na zbiorze walidacyjnym
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("Model saved.")
    
    print("Training finished.")


if __name__ == "__main__":
    merged_data, bounds = consolidate_data(
        "data/2024-04-12_15-35-48-dobre.pickle",
        "data/2024-04-12_20-57-44-n.pickle",
        "data/2024-04-12_21-12-06-n.pickle",
        "data/2024-04-14_15-28-57-n.pickle"
    )
    data = delete_disturbing_data(merged_data, bounds)

    attributes, labels = generate_attributes(data, bounds)
    attributes_ndarray = np.array(attributes)
    labels_ndarray = np.array(labels)

    X = attributes_ndarray.tolist()  # Dane wejściowe
    y = labels_ndarray.tolist()  # Etykiety
    y = [direction.value for direction in y]

    # Podział na zbiór treningowy (80%) i pozostałe dane (20%)
    X_train, X_remain, y_train, y_remain = train_test_split(X, y, test_size=0.2, random_state=42)

    # Podział pozostałych danych na zbiory walidacyjny i testowy (po 50%)
    X_val, X_test, y_val, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=42)

    input_size = len(X_train[0])  # Rozmiar wejścia to liczba cech w jednej próbce danych
    hidden_size = 64 
    output_size = 4  
    num_epochs = 5
    learning_rate = 0.001
    batch_size = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path = 'best_model.pth'

    train_dataset = BCDataset(X_train, y_train)
    val_dataset = BCDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # for batch in dataloader:
    #     batch_data = batch['data']
    #     batch_labels = batch['label']
    #     print("Batch data:", batch_data)
    #     print("Batch labels:", batch_labels)

   

    # Inicjalizacja modelu MLP
    model = MLP(input_size, hidden_size, output_size).to(device)

    # Tworzenie obiektów DataLoader dla zbiorów treningowego i walidacyjnego
    # train_dataset = BCDataset(X_train, y_train)
    # val_dataset = BCDataset(X_val, y_val)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Definicja funkcji straty (kosztu) i optymalizatora
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Trenowanie modelu
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_path)





'''
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

'''