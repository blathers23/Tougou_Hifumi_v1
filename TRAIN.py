import MCTS
import EGAME as env
import random
import numpy as np
import time
import pandas as pd
import math
import copy

Bboardvalue = np.zeros([5,5])
Bboardvalue[0][0] = 100
Bboardvalue[0][1] = 12
Bboardvalue[1][0] = 12   
Bboardvalue[1][1] = 8
Bboardvalue[0][2] = 6
Bboardvalue[2][0] = 6
Bboardvalue[2][1] = 4
Bboardvalue[1][2] = 4
Bboardvalue[2][2] = 4
Bboardvalue[0][3] = 3
Bboardvalue[3][0] = 3
Bboardvalue[3][1] = 2
Bboardvalue[3][2] = 2
Bboardvalue[3][3] = 2
Bboardvalue[2][3] = 2
Bboardvalue[1][3] = 2
Bboardvalue[4][0] = 1
Bboardvalue[4][1] = 1
Bboardvalue[4][2] = 1
Bboardvalue[4][3] = 1
Bboardvalue[3][4] = 1
Bboardvalue[2][4] = 1
Bboardvalue[1][4] = 1
Bboardvalue[0][4] = 1
Bboardvalue[4][4] = 0
Rboardvalue = np.flip(Bboardvalue)
Game = copy.deepcopy(env.Game())

METHOD = ['UCT','MCTS'][0]
UCTSTEPS = 10000
MCTSSTEPS = 3000

def onemain():
    Game.RandResetBoard()
    Player = random.choice([-1, 1])
    while True:
        Rand = random.randint(1, 6) * Player
        position0, position1 = Game.ChooseChess(Rand)
        if position0 == position1:
            MoveDirection = GetMove(Game.board, position0)
            Memory(position0, MoveDirection)
            Game.Move(position0, MoveDirection)
        else:
            MoveDirection0 = GetMove(Game.board, position0)
            MoveDirection1 = GetMove(Game.board, position1)
            if Score(Game.board, position0, MoveDirection0, position1, MoveDirection1) == 0:
                Memory(position0, MoveDirection0)
                Game.Move(position0, MoveDirection0)
            else:
                Memory(position1, MoveDirection1)
                Game.Move(position1, MoveDirection1)
        Winner = Game.GetWinner()
        print(Game.board)
        print(' #####################')
        #time.sleep(2)
        if Winner != 0:
            break
        Player = -Player

def main():
    while True:
        Game.RandResetBoard()
        Player = random.choice([-1, 1])
        while True:
            Rand = random.randint(1, 6) * Player
            position0, position1 = Game.ChooseChess(Rand)
            if position0 == position1:
                MoveDirection = GetMove(Game.board, position0)
                Memory(position0, MoveDirection)
                Game.Move(position0, MoveDirection)
            else:
                MoveDirection0 = GetMove(Game.board, position0)
                MoveDirection1 = GetMove(Game.board, position1)
                if Score(Game.board, position0, MoveDirection0, position1, MoveDirection1) == 0:
                    Memory(position0, MoveDirection0)
                    Game.Move(position0, MoveDirection0)
                else:
                    Memory(position1, MoveDirection1)
                    Game.Move(position1, MoveDirection1)
            Winner = Game.GetWinner()
            print(Game.board)
            print(' #####################')
            #time.sleep(2)
            if Winner != 0:
                break
            Player = -Player

def GetMove(board, position):
    if board[position[0]][position[1]] > 0:
        if position[0] == 4:
            Move = 0
        elif position[1] == 4:
            Move = 1
        else:
            if METHOD == 'MCTS':
                WinRate0 = MCTS.MCTS(board.copy(), position, 0, STEPS = MCTSSTEPS)
                WinRate1 = MCTS.MCTS(board.copy(), position, 1, STEPS = MCTSSTEPS)
                WinRate2 = MCTS.MCTS(board.copy(), position, 2, STEPS = MCTSSTEPS)
                Move = np.argmax([WinRate0, WinRate1, WinRate2])
            elif METHOD == 'UCT':
                Move = UCTMove(board.copy(), position)
    elif board[position[0]][position[1]] < 0:
        if position[0] == 0:
            Move = 0
        elif position[1] == 0:
            Move = 1
        else:
            if METHOD == 'MCTS':
                WinRate0 = MCTS.MCTS(board.copy(), position, 0, STEPS = MCTSSTEPS)
                WinRate1 = MCTS.MCTS(board.copy(), position, 1, STEPS = MCTSSTEPS)
                WinRate2 = MCTS.MCTS(board.copy(), position, 2, STEPS = MCTSSTEPS)
                Move = np.argmax([WinRate0, WinRate1, WinRate2])
            elif METHOD == 'UCT':
                Move = UCTMove(board.copy(), position)

    return Move

def Score(board, position0, move0, position1, move1):
    if board[position0[0]][position0[1]] > 0:
        if move0 == 0:
            attack0 = Rboardvalue[position0[0]][position0[1] + 1] - GetPD(board, [position0[0], position0[1] + 1])
        elif move0 == 1:
            attack0 = Rboardvalue[position0[0] + 1][position0[1]] - GetPD(board, [position0[0] + 1, position0[1]])
        elif move0 == 2:
            attack0 = Rboardvalue[position0[0] + 1][position0[1] + 1] - GetPD(board, [position0[0] + 1, position0[1] + 1])
        if move1 == 0:
            attack1 = Rboardvalue[position1[0]][position1[1] + 1] - GetPD(board, [position1[0], position1[1] + 1])
        elif move1 == 1:
            attack1 = Rboardvalue[position1[0] + 1][position1[1]] - GetPD(board, [position1[0] + 1, position1[1]])
        elif move1 == 2:
            attack1 = Rboardvalue[position1[0] + 1][position1[1] + 1] - GetPD(board, [position1[0] + 1, position1[1] + 1])
    elif board[position0[0]][position0[1]] < 0:
        if move0 == 0:
            attack0 = Bboardvalue[position0[0]][position0[1] - 1] - GetPD(board, [position0[0], position0[1] - 1])
        elif move0 == 1:
            attack0 = Bboardvalue[position0[0] - 1][position0[1]] - GetPD(board, [position0[0] - 1, position0[1]])
        elif move0 == 2:
            attack0 = Bboardvalue[position0[0] - 1][position0[1] - 1] - GetPD(board, [position0[0] - 1, position0[1] - 1])
        if move1 == 0:
            attack1 = Bboardvalue[position1[0]][position1[1] - 1] - GetPD(board, [position1[0], position1[1] - 1])
        elif move1 == 1:
            attack1 = Bboardvalue[position1[0] - 1][position1[1]] - GetPD(board, [position1[0] - 1, position1[1]])
        elif move1 == 2:
            attack1 = Bboardvalue[position1[0] - 1][position1[1] - 1] - GetPD(board, [position1[0] - 1, position1[1] - 1])

    return np.argmax([attack0, attack1])

def GetPD(board, position, sum = 0):
    if board[position[0]][position[1]] > 0:
        for Randnum in [1,2,3,4,5,6]:
            P0, P1 = Game.ChooseChess(Randnum)
            if P0 == position or P1 == position:
                sum += 1 
        PD = sum/6 * max(position)
    else:
        for Randnum in [-1,-2,-3,-4,-5,-6]:
            P0, P1 = Game.ChooseChess(Randnum)
            if P0 == position or P1 == position:
                sum += 1
        PD = sum/6 * max(position)

    return PD

def UCTMove(Mboard, position):
    N = [0, 0, 0]
    UCTvalue = [0, 0, 0]
    WinSum = [0, 0, 0]
    for Step in range(UCTSTEPS):
        board = Mboard.copy()
        i = np.argmax(UCTvalue)
        WinSum[i] += MCTS.MCTS(board, position, i, STEPS = 1)
        N[i] += 1
        UCTvalue[0] = WinSum[0] / (N[0] + 1e-99) + (2 * math.log(Step + 1)/(N[0] + 1e-99)) ** 0.5
        UCTvalue[1] = WinSum[1] / (N[1] + 1e-99) + (2 * math.log(Step + 1)/(N[1] + 1e-99)) ** 0.5
        UCTvalue[2] = WinSum[2] / (N[2] + 1e-99) + (2 * math.log(Step + 1)/(N[2] + 1e-99)) ** 0.5
    print('WinSum:', WinSum)
    print('     N:', N)

    return np.argmax(np.array(WinSum) / np.array(N))


def ChangeBoard():
    TempBoard = Game.board.copy()
    TempBoard = np.where(TempBoard < 0, -TempBoard + 6, TempBoard)
    
    return TempBoard

def FlipChangeBoard():
    TempBoard = Game.board.copy()
    TempBoard = np.flip(-TempBoard)
    TempBoard = np.where(TempBoard < 0, -TempBoard + 6, TempBoard)
    
    return TempBoard

def Memory(position, movedirection):
    Board = ChangeBoard()
    FlipBoard = FlipChangeBoard()
    Board[position[0]][position[1]] = -Board[position[0]][position[1]]
    outputchess = pd.DataFrame(Board.reshape(25)) / 12
    TempMove = np.zeros(3)
    TempMove[movedirection] = 1.
    ready = outputchess.append(pd.DataFrame(TempMove))
    ready = ready.T
    ready.to_csv('board0.csv', header = 0, index = 0, mode = 'a')
    FlipBoard[-position[0] + 4][-position[1] + 4] = -FlipBoard[-position[0] + 4][-position[1] + 4]
    Flipoutputchess = pd.DataFrame(FlipBoard.reshape(25)) / 12
    Flipready = Flipoutputchess.append(pd.DataFrame(TempMove))
    Flipready = Flipready.T
    Flipready.to_csv('board0.csv', header = 0, index = 0, mode = 'a')

main()