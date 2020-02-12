import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import VEGAME as env
import random
import numpy as np
import time
import pandas as pd
import copy
import tensorflow as tf
import a_forward
import a_backward

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

Game = copy.deepcopy(env.Game(np.zeros([5,5])))
sess = tf.Session()
tf.Graph().as_default()
x = tf.placeholder(tf.float32,[
    1,
    a_forward.BOARD_SIZE, 
    a_forward.BOARD_SIZE,
    a_forward.NUM_CHANNELS])#重现计算图
y = a_forward.forward(x,False,None)#计算求得y
preValue = tf.argmax(y,1)#y的最大值对应的列表索引号即为最大值

variable_averages = tf.train.ExponentialMovingAverage(a_backward.MOVING_AVERAGE_DECAY)
variables_to_restore = variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)#实例化带有滑动平均值的saver

ckpt = tf.train.get_checkpoint_state(a_backward.MODEL_SAVE_PATH)#用with结构加载ckpt
saver.restore(sess, ckpt.model_checkpoint_path)

def MCTS(board, position, direction, WINSUM = 0, STEPS = 3000, is_TF = True):
    global Game
    Game = copy.deepcopy(env.Game(board))
    if board[position[0]][position[1]] > 0:
        APlayer = 1
    elif board[position[0]][position[1]] < 0:
        APlayer = -1
    for _ in range(STEPS):
        Game.ResetBoard()
        Game.Move(position, direction)
        Winner = Game.GetWinner()
        if Winner == APlayer:
            WINSUM += 1
        if Winner != 0:
            break
        Player = - APlayer
        while True:
            Rand = random.randint(1, 6) * Player
            position0, position1 = Game.ChooseChess(Rand)
            if position0 == position1:
                MoveDirection = GetMove(Game.board, position0, is_TF)
                Game.Move(position0, MoveDirection)
            else:
                MoveDirection0 = GetMove(Game.board, position0, is_TF)
                MoveDirection1 = GetMove(Game.board, position1, is_TF)
                if Score(Game.board, position0, MoveDirection0, position1, MoveDirection1) == 0:
                    Game.Move(position0, MoveDirection0)
                else:
                    Game.Move(position1, MoveDirection1)

            Winner = Game.GetWinner()
            
            if Winner == APlayer:
                WINSUM += 1

            if Winner != 0:
                break

            Player = - Player

    return WINSUM / STEPS

def GetMove(board, position, is_TF):
    if board[position[0]][position[1]] > 0:
        if position[0] == 4:
            Move = 0
        elif position[1] == 4:
            Move = 1
        else:
            if is_TF == True:
                Move = AIMove(position)
            else:
                Move = random.randint(0, 2)
    elif board[position[0]][position[1]] < 0:
        if position[0] == 0:
            Move = 0
        elif position[1] == 0:
            Move = 1
        else:
            if is_TF == True:
                Move = AIMove(position)
            else:
                Move = random.randint(0, 2)

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

def ChangeBoard():
    global Game
    TempBoard = Game.board.copy()
    TempBoard = np.where(TempBoard < 0, -TempBoard + 6, TempBoard)

    return TempBoard

def AIMove(position, Board = ChangeBoard()):
    Board[position[0]][position[1]] = -Board[position[0]][position[1]]
    board_ready = np.reshape(Board,(
            1,
            a_forward.BOARD_SIZE,
            a_forward.BOARD_SIZE,
            a_forward.NUM_CHANNELS))
    Move = sess.run(preValue, feed_dict={x:board_ready})
    
    return Move
