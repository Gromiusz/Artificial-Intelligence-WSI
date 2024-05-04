import pickle
from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def int_to_direction(num):
    if num == 0:
        return Direction.UP
    elif num == 1:
        return Direction.RIGHT
    elif num == 2:
        return Direction.DOWN
    elif num == 3:
        return Direction.LEFT
    else:
        raise ValueError("Numer musi być w zakresie od 0 do 3")
    

def move(body, direction, block_size, length):
    copy_of_body = body.copy()
    curr_head = copy_of_body[-1]
    if direction == Direction.DOWN:
        next_head = (curr_head[0], curr_head[1] + block_size)
        copy_of_body.append(next_head)
    elif direction == Direction.UP:
        next_head = (curr_head[0], curr_head[1] - block_size)
        copy_of_body.append(next_head)
    elif direction == Direction.RIGHT:
        next_head = (curr_head[0] + block_size, curr_head[1])
        copy_of_body.append(next_head)
    elif direction == Direction.LEFT:
        next_head = (curr_head[0] - block_size, curr_head[1])
        copy_of_body.append(next_head)

    if length < len(copy_of_body):
        copy_of_body.pop(0)
    return copy_of_body


def is_tail_collision(body):
    head = body[-1]
    has_eaten_tail = False

    for i in range(len(body) - 1):
        segment = body[i]
        if head[0] == segment[0] and head[1] == segment[1]:
            has_eaten_tail = True

    return has_eaten_tail


def is_wall_collision(snake_body, onedata_direction, bounds):
    up_die_condition = (snake_body[-1][1] == 0) and (onedata_direction == Direction.UP.value)
    right_die_condition = snake_body[-1][0] == bounds[0] - 30 and onedata_direction == Direction.RIGHT.value
    down_die_condition = snake_body[-1][1] == bounds[1] - 30 and onedata_direction == Direction.DOWN.value
    left_die_condition = snake_body[-1][0] == 0 and onedata_direction == Direction.LEFT.value
    return up_die_condition or right_die_condition or down_die_condition or left_die_condition
    

def read_data_and_bounds(file_path):
    print("Reading data...", end = ' ')
    with open(file_path, 'rb') as file:
        data_file = pickle.load(file)
    data = data_file["data"]
    bounds = data_file["bounds"]
    print("Done", end='   ')
    print(f"(Data length: {len(data)})")
    return data, bounds


def consolidate_data(*file_paths):
    data = [None] * len(file_paths)
    bounds = [None] * len(file_paths)

    for i, file_path in enumerate(file_paths):
        data[i], bounds[i] = read_data_and_bounds(file_path)

    print("Consolidating data...", end = ' ')
    merged_data = data[0].copy()
    for i in range(len(file_paths) - 1):
        merged_data.extend(data[i + 1])
        if not bounds[i] == bounds[i + 1]:
            raise ValueError("BoundsNotEqual")

    equal_bounds = bounds[0]
    
    print("Done", end='   ')
    print(f"(Data length: {len(merged_data)})")

    return merged_data, equal_bounds


def delete_disturbing_data(data, bounds):
    new_data =[]
    print("Deleting disturbing data...", end = ' ')
    for onedata in data:
        # print(onedata)
        snake = onedata[0]
        food = snake['food']
        snake_body = snake['snake_body']
        onedata_direction = onedata[1].value

        onedata_direction_enum = int_to_direction(onedata_direction)
        copy_of_body = move(snake_body, onedata_direction_enum, 30, len(snake_body)) # wyznaczenie pozycji węża po wykonaniu ruchu w zadanym kierynku w celu sprawdzenia czy w najbliższej przyszłości nie będzie kolizji

        if not (is_wall_collision(snake_body, onedata_direction, bounds) or is_tail_collision(copy_of_body)):
            new_data.append(onedata)
    print("Done")
    return new_data

if __name__ == "__main__":
    merged_data, bounds = consolidate_data(
        "data/2024-04-12_19-35-18-short_test.pickle",
        "data/2024-04-14_11-00-40-short.pickle"
        )
    data = delete_disturbing_data(merged_data, bounds)
    for onedata in data:
        print(onedata)


'''
print(data[0])
print("\n")
print(data[1])
print("\n")
print(merged_data)


# data = ({'food': (270, 270), 'snake_body': [(30, 0), (30, 30), (30, 60)], 'snake_direction': <Direction.DOWN: 2>}, <Direction.DOWN: 2>)
with open(f"data/2024-04-12_19-35-18-short_test.pickle", 'rb') as f:
    data_file = pickle.load(f)


data = data_file["data"]
bounds = data_file["bounds"]

i=0
newSnakeLength = 0
new_data =[]
for onedata in data:
    print(onedata)
    snake = onedata[0]
#     # print(snake)
    food = snake['food']
#     # print(food)
    snake_body = snake['snake_body']
    # onedata_direction = Direction(onedata[1])
#     # print(snake_body[0])
#     # print(len(snake_body))
#     oldSnakeLength = newSnakeLength
#     newSnakeLength = len(snake_body)
    onedata_direction = onedata[1].value
    
    # print((snake_body[-1][1] == 0) and (onedata[1] == Direction.UP))
    # print((snake_body[-1][1] == 0))
    # print(onedata_direction == Direction.UP.value)
    # print(onedata[1])
    # print(onedata_direction == 0)

    
    # up_die_condition = (snake_body[-1][1] == 0) and (onedata_direction == Direction.UP.value)
    # right_die_condition = snake_body[-1][0] == bounds[0] - 30 and onedata_direction == Direction.RIGHT.value
    # down_die_condition = snake_body[-1][1] == bounds[1] - 30 and onedata_direction == Direction.DOWN.value
    # left_die_condition = snake_body[-1][0] == 0 and onedata_direction == Direction.LEFT.value

    onedata_direction_enum = int_to_direction(onedata_direction)

    copy_of_body = move(snake_body, onedata_direction_enum, 30, len(snake_body)) # wyznaczenie pozycji węża po wykonaniu ruchu w zadanym kierynku w celu sprawdzenia czy w najbliższej przyszłości nie będzie kolizji

    # print(snake_body[-1][1])

    # if not (up_die_condition or right_die_condition or down_die_condition or left_die_condition or is_tail_collision(copy_of_body)):
    if not (is_wall_collision(snake_body, onedata_direction, bounds) or is_tail_collision(copy_of_body)):
        new_data.append(onedata)

# #     if(newSnakeLength<oldSnakeLength):
# #         data[i-1]=onedata
# #         i -= 1
#     # if i==5:
#     #     break
#     i+=1
print("\n")
for onedata in new_data:
    print(onedata)


'''