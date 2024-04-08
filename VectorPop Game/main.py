import random

random.seed(304260%42)  # TODO: For final results set seed as your student's id modulo 42


class RandomAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if random.random() > 0.5:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class GreedyAgent:
    def __init__(self):
        self.numbers = []

    def act(self, vector: list):
        if vector[0] > vector[-1]:
            self.numbers.append(vector[0])
            return vector[1:]
        self.numbers.append(vector[-1])
        return vector[:-1]


class NinjaAgent:
    """   ⠀⠀⠀⠀⠀⣀⣀⣠⣤⣀⣀⣀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣤⣀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠴⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠠⠶⠶⠶⠶⢶⣶⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⠀⠀⠀
⠀⠀⠀⠀⢀⣴⣶⣶⣶⣶⣶⣶⣦⣬⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
⠀⠀⠀⠀⣸⣿⡿⠟⠛⠛⠋⠉⠉⠉⠁⠀⠀⠀⠈⠉⠉⠉⠙⠛⠛⠿⣿⣿⡄⠀
⠀⠀⠀⠀⣿⠋⠀⠀⠀⠐⢶⣶⣶⠆⠀⠀⠀⠀⠀⢶⣶⣶⠖⠂⠀⠀⠈⢻⡇⠀
⠀⠀⠀⠀⢹⣦⡀⠀⠀⠀⠀⠉⢁⣠⣤⣶⣶⣶⣤⣄⣀⠀⠀⠀⠀⠀⣀⣾⠃⠀
⠀⠀⠀⠀⠘⣿⣿⣿⣶⣶⣶⣾⣿⣿⣿⡿⠿⠿⣿⣿⣿⣿⣷⣶⣾⣿⣿⡿⠀⠀
⠀⠀⢀⣴⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀
⠀⠀⣾⡿⢃⡀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀
⠀⢸⠏⠀⣿⡇⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀
⠀⠀⠀⢰⣿⠃⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⠛⠛⣉⣁⣤⡶⠁⠀⠀⠀⠀⠀
⠀⠀⣠⠟⠁⠀⠀⠀⠀⠀⠈⠛⠿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠁⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠛⠉⠉⠁⠀⠀⠀⠀⠀⠀⠀
                かかって来い! """
    def __init__ (OOOO000O000O00000 ):
        OOOO000O000O00000 .numbers =[]
    def act (O000000O000OO0O0O ,O0OO0O0O0O0OO0O00 ):
        if len (O0OO0O0O0O0OO0O00 )%2 ==0 :
            O00O0O0000000OO0O =sum (O0OO0O0O0O0OO0O00 [::2 ])
            O0O00O0OO00O0O0O0 =sum (O0OO0O0O0O0OO0O00 )-O00O0O0000000OO0O
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
                return O0OO0O0O0O0OO0O00 [1 :] # explained: https://r.mtdv.me/articles/k1evNIASMp
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
            return O0OO0O0O0O0OO0O00 [:-1 ]
        else :
            O00O0O0000000OO0O =max (sum (O0OO0O0O0O0OO0O00 [1 ::2 ]),sum (O0OO0O0O0O0OO0O00 [2 ::2 ]))
            O0O00O0OO00O0O0O0 =max (sum (O0OO0O0O0O0OO0O00 [:-1 :2 ]),sum (O0OO0O0O0O0OO0O00 [:-2 :2 ]))
            if O00O0O0000000OO0O >=O0O00O0OO00O0O0O0 :
                O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [-1 ])
                return O0OO0O0O0O0OO0O00 [:-1 ]
            O000000O000OO0O0O .numbers .append (O0OO0O0O0O0OO0O00 [0 ])
            return O0OO0O0O0O0OO0O00 [1 :]


class MinMaxAgent:
    def __init__(self, max_depth=5):
        self.numbers = []
        self.max_depth = max_depth
    
    def act(self, vector: list):
        analyzed_results = self.minimax(vector, (0, 0), 1) # otrzymujemy wektor ze wszystkimi możliwościami rozgrywki
        moveRight = self.find_best_move(analyzed_results) 

        if(moveRight):
            self.numbers.append(vector[-1])
            return vector[:-1]
        self.numbers.append(vector[0])
        return vector[1:]

    def minimax(self, currentVector, moves, player, depth = 0):
        if len(currentVector) == 0 or depth - self.max_depth == 0:
            return [(moves[0], moves[1])] # wektor przechowije sumę punktów odpowiednio dla gracza 1 i 2

        if player == 1:
            return self.minimax(currentVector[1:], (moves[0] + currentVector[0], moves[1]), 2, depth+1) + \
                   self.minimax(currentVector[:-1], (moves[0] + currentVector[-1], moves[1]), 2, depth+1) 
        else:
            return self.minimax(currentVector[1:], (moves[0], moves[1] + currentVector[0]), 1, depth+1) + \
                   self.minimax(currentVector[:-1], (moves[0], moves[1] + currentVector[-1]), 1, depth+1)
        
    def find_best_move(self, input_list: list):
        all_moves = [input_list[i*2] for i in range(len(input_list)//2)]
        whoWin = [1 if i[0]>i[1] else 0 for i in all_moves] # przypisanie 1 jeśli wygrał player nr 1

        sum1 = sum(whoWin[:len(whoWin)//2])
        sum2 = sum(whoWin[len(whoWin)//2:])
        # wybierany jest prawa lub lewa strona w zależności od tego po której stronie jest więcej zwycięstw
        moveRight = True if sum1<sum2 else False
        return moveRight

    
def run_game(vector, first_agent, second_agent):
    while len(vector) > 0:
        vector = first_agent.act(vector)
        if len(vector) > 0:
            vector = second_agent.act(vector)


def main():
    vector = [random.randint(-10, 10) for _ in range(14)]
    print(f"Vector: {vector}")
    first_agent, second_agent = MinMaxAgent(), GreedyAgent()
    run_game(vector, first_agent, second_agent)

    print(f"First agent: {sum(first_agent.numbers)} Second agent: {sum(second_agent.numbers)}\n"
          f"First agent: {first_agent.numbers}\n"
          f"Second agent: {second_agent.numbers}")


if __name__ == "__main__":
    main()
