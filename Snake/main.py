import copy
import os
import pickle
import pygame
import time

from food import Food
from model import game_state_to_data_sample
from model import get_predicted_move
from snake import Snake, Direction
from statistics import mean


def main():
    print("Initializating game...", end=' ')
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)
    print("Done")
    #agent = HumanAgent(block_size, bounds)  # Once your agent is good to go, change this line
    agent = BehavioralCloningAgent(block_size, bounds)  # Once your agent is good to go, change this line
    scores = []
    run = True
    run_count = 0
    pygame.time.delay(1000)
    # for _ in range(100):# while run:
    # while run_count<100:
    while run:
        pygame.time.delay(60)  # Adjust game speed, decrease to test your agent and model quickly ori=60

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            # pygame.time.delay(2)
            scores.append(snake.length - 3)
            run_count += 1
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    print(f"Mean scores: {mean(scores)}")
    agent.dump_data()
    pygame.quit()


class HumanAgent:
    """ In every timestep every agent should perform an action (return direction) based on the game state. Please note, that
    human agent should be the only one using the keyboard and dumping data. """
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        folder_path = "data"
        os.makedirs(folder_path, exist_ok=True) 

        current_time = time.strftime('%Y-%m-%d_%H-%M-%S')  
        file_path = os.path.join(folder_path, f"{current_time}.pickle")

        try:
            with open(file_path, 'wb') as f:
                pickle.dump({"block_size": self.block_size,
                             "bounds": self.bounds,
                             "data": self.data[:-10]}, f) 
            print("Data has been successfuly saved.")
        except Exception as e:
            print(f"Error while saving data: {e}")



class BehavioralCloningAgent:
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""

        keys = get_predicted_move(game_state)
        # print(keys)
        action = game_state["snake_direction"]
        if keys == 3:
            action = Direction.LEFT
        elif keys == 1:
            action = Direction.RIGHT
        elif keys == 0:
            action = Direction.UP
        elif keys == 2:
            action = Direction.DOWN
        return action

    def dump_data(self):
        pass


if __name__ == "__main__":
    main()

