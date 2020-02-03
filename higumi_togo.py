#coding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
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
saver.restore(sess,ckpt.model_checkpoint_path)

class EinStein_Game():
    def __init__(self):
        self.board = np.zeros([5,5])
        self.PI = []
        self.mode = ['game', 'test'][1]
        self.Player = 0

    def main(self):
        while True:
            print('#########################')
            print('#######Higumi_Togo#######')
            print('#########################')
            print()
            Tip1 = input('请输入"N"开始新游戏, 输入"S"保存上一局棋谱：')
            if Tip1 == "N":
                Tip2 = input('人人对弈请输入"H", 人机对弈请输入"A":')
                
                if Tip2 == "H":
                    redBoard = input('请输入红方(左上方)棋子布局,以逗号间隔：')
                    redBoard = [float(n) for n in redBoard.split(',')]
                    blueBoard = input('请输入蓝方（右下方）棋子布局,以逗号间隔：')
                    blueBoard = [-float(n) for n in blueBoard.split(',')]
                    self.Player = float(input('请输入先手方：'))
                    self.createBoard(redBoard, blueBoard)
                    self.PI = []
                    self.PI.append(list(self.board.reshape(25)))
                    while True:
                        os.system('cls')
                        print(self.board)
                        if self.Player > 0:
                            print('#####红方时间#####')
                        else:
                            print('*****蓝方时间*****')
                        rand = int(input('请输入随机数,输入0以悔棋：'))
                        if rand == '0':
                            if len(self.PI) == 0:
                                print('爪巴！赶紧的')
                            else:
                                self.Player = -self.Player
                                del self.PI[len(self.PI) - 1]
                                self.board = np.array(self.PI[len(self.PI) - 1])
                                _ = input('退回上一局面，按回车以继续')
                                continue
                        if self.Player * rand < 0:
                            print('!!!!!WARNING!!!!!')
                        chessPosition0, chessPosition1 = self.ChooseChess(rand)
                        Chess = int(input('请输入移动的棋子：'))
                        if Chess not in [self.board[chessPosition0[0], chessPosition0[1]], self.board[chessPosition1[0], chessPosition1[1]]]:
                            print('!!!!!WARNING!!!!!')
                            while True:
                                Chess = int(input('请输入移动的棋子：'))
                                if Chess in [self.board[chessPosition0[0], chessPosition0[1]], self.board[chessPosition1[0], chessPosition1[1]]]:
                                    break
                                else:
                                    print('!!!!!WARNING!!!!!')
                        if Chess == self.board[chessPosition0[0], chessPosition0[1]]:
                            position = chessPosition0
                        else:
                            position = chessPosition1
                        moveDirection = self.GetMove(position)
                        self.Move(position, moveDirection)
                        self.PI.append(list(self.board.reshape(25)))
                        winner = self.GetWinner()
                        if winner != 0:
                            os.system('cls')
                            if winner == 1:
                                print('红方获胜')
                            else:
                                print('蓝方获胜')
                            _ = input('回车以继续')
                            os.system('cls')
                            break
                        self.Player = -self.Player
                
                elif Tip2 == 'A':
                    redBoard = input('请输入红方(左上方)棋子布局,以逗号间隔：')
                    redBoard = [float(n) for n in redBoard.split(',')]
                    blueBoard = input('请输入蓝方（右下方）棋子布局,以逗号间隔：')
                    blueBoard = [-float(n) for n in blueBoard.split(',')]
                    
                    if self.mode == 'test':
                        self.Player = float(input('请输入先手方：'))
                        self.createBoard(redBoard, blueBoard)
                        self.PI = []
                        self.PI.append(list(self.board.reshape(25)))
                        while True:
                            os.system('cls')
                            print(self.board)
                            if self.Player > 0:
                                print('#####红方时间#####')
                            else:
                                print('*****蓝方时间*****')
                            rand = int(input('请输入随机数,输入0以悔棋：'))
                            if rand == '0':
                                if len(self.PI) == 0:
                                    print('爪巴！赶紧的')
                                else:
                                    self.Player = -self.Player
                                    del self.PI[len(self.PI) - 1]
                                    self.board = np.array(self.PI[len(self.PI) - 1])
                                    _ = input('退回上一局面，按回车以继续')
                                    continue
                            print('随机数为：', rand)
                            if self.Player * rand < 0:
                                print('!!!!!WARNING!!!!!')
                            chessPosition0, chessPosition1 = self.ChooseChess(rand)
                            Tip4 = input('人走棋请输入"H", AI走棋请输入"A":')
                            if Tip4 == "H":
                                Chess = int(input('请输入移动的棋子：'))
                                if Chess not in [self.board[chessPosition0[0], chessPosition0[1]], self.board[chessPosition1[0], chessPosition1[1]]]:
                                    print('!!!!!WARNING!!!!!')
                                    while True:
                                        Chess = int(input('请输入移动的棋子：'))
                                        if Chess in [self.board[chessPosition0[0], chessPosition0[1]], self.board[chessPosition1[0], chessPosition1[1]]]:
                                            break
                                        else:
                                            print('!!!!!WARNING!!!!!')
                                if Chess == self.board[chessPosition0[0], chessPosition0[1]]:
                                    position = chessPosition0
                                else:
                                    position = chessPosition1
                                moveDirection = self.GetMove(position)
                            elif Tip4 == "A":
                                print('AI落子思考中...')
                                time_start = time.time()
                                #position = "?"
                                #moveDirection = "?"
                                position, moveDirection = self.AIMove(chessPosition0, chessPosition1)
                                time_end = time.time()
                                print('AI思考用时为: ', time_end-time_start, ' 秒.')
                            self.Move(position, moveDirection)
                            self.PI.append(list(self.board.reshape(25)))
                            winner = self.GetWinner()
                            if winner != 0:
                                os.system('cls')
                                if winner == 1:
                                    print('红方获胜')
                                else:
                                    print('蓝方获胜')
                                _ = input('回车以继续')
                                os.system('cls')
                                break
                            self.Player = -self.Player

                    elif self.mode == 'game':
                        self.Player = float(input('请输入先手方：'))
                        AIPlayer = float(input('请输入AI方：'))
                        self.createBoard(redBoard, blueBoard)
                        self.PI = []
                        self.PI.append(list(self.board.reshape(25)))
                        while True:
                            os.system('cls')
                            print(self.board)
                            if self.Player > 0:
                                print('#####红方时间#####')
                            else:
                                print('*****蓝方时间*****')
                            rand = int(input('请输入随机数,输入0以悔棋：'))
                            if rand == '0':
                                if len(self.PI) == 0:
                                    print('爪巴！赶紧的')
                                else:
                                    self.Player = -self.Player
                                    del self.PI[len(self.PI) - 1]
                                    self.board = np.array(self.PI[len(self.PI) - 1])
                                    _ = input('退回上一局面，按回车以继续')
                                    continue
                            print('随机数为：', rand)
                            if self.Player * rand < 0:
                                print('!!!!!WARNING!!!!!')
                            chessPosition0, chessPosition1 = self.ChooseChess(rand)
                            if AIPlayer != self.Player:
                                Chess = int(input('请输入移动的棋子：'))
                                if Chess not in [self.board[chessPosition0[0], chessPosition0[1]], self.board[chessPosition1[0], chessPosition1[1]]]:
                                    print('!!!!!WARNING!!!!!')
                                    while True:
                                        Chess = int(input('请输入移动的棋子：'))
                                        if Chess in [self.board[chessPosition0[0], chessPosition0[1]], self.board[chessPosition1[0], chessPosition1[1]]]:
                                            break
                                        else:
                                            print('!!!!!WARNING!!!!!')
                                if Chess == self.board[chessPosition0[0], chessPosition0[1]]:
                                    position = chessPosition0
                                else:
                                    position = chessPosition1
                                moveDirection = self.GetMove(position)
                            elif AIPlayer == self.Player:
                                print('AI落子思考中...')
                                time_start = time.time()
                                #position = "?"
                                #moveDirection = "?"
                                position, moveDirection = self.AIMove(chessPosition0, chessPosition1)
                                time_end = time.time()
                                print('AI思考用时为: ', time_end-time_start, ' 秒.')
                            self.Move(position, moveDirection)
                            self.PI.append(list(self.board.reshape(25)))
                            winner = self.GetWinner()
                            if winner != 0:
                                os.system('cls')
                                if winner == 1:
                                    print('红方获胜')
                                else:
                                    print('蓝方获胜')
                                _ = input('回车以继续')
                                os.system('cls')
                                break
                            self.Player = -self.Player
            elif Tip1 == "S":
                self.Saver()
                print('提示：棋谱保存成功')
                _ = input('回车以继续：')
            os.system('cls')

    def GetMove(self, position):
        if self.board[position[0]][position[1]] > 0:
            if position[0] == 4:
                Move = 0
            elif position[1] == 4:
                Move = 1
            else:
                Move = int(input('请输入移动方向(0-平,1-竖,2-斜)：'))
                #Move = random.randint(0, 2)
        elif self.board[position[0]][position[1]] < 0:
            if position[0] == 0:
                Move = 0
            elif position[1] == 0:
                Move = 1
            else:
                Move = int(input('请输入移动方向(0-平,1-竖,2-斜)：'))
                #Move = random.randint(0, 2)
        
        return Move
                    
    def createBoard(self, red, blue):
        self.board = np.zeros([5,5])
        self.board[0][0] =  red[0]
        self.board[0][1] =  red[1]
        self.board[0][2] =  red[2]
        self.board[1][0] =  red[3]
        self.board[1][1] =  red[4]
        self.board[2][0] =  red[5]
        self.board[2][4] = blue[0]
        self.board[3][3] = blue[1]
        self.board[3][4] = blue[2]
        self.board[4][2] = blue[3]
        self.board[4][3] = blue[4]
        self.board[4][4] = blue[5]
        #print(self.board)

    def ChooseChess(self, Randnum):
        tempboard = self.board.copy()
        if Randnum > 0:
            tempboard = np.where(tempboard < 0, 0, tempboard)
        elif Randnum < 0:
            tempboard = np.where(tempboard > 0, 0, tempboard)
            Randnum = -Randnum
            tempboard = -tempboard
        tempboard -= Randnum
        #print(tempboard)
        if 0 in tempboard:
            return [np.where(tempboard == 0)[0][0], np.where(tempboard == 0)[1][0]], [np.where(tempboard == 0)[0][0], np.where(tempboard == 0)[1][0]]
        else:
            if (tempboard < 0).all():
                return [np.where(tempboard == tempboard.max())[0][0], np.where(tempboard == tempboard.max())[1][0]], [np.where(tempboard == tempboard.max())[0][0], np.where(tempboard == tempboard.max())[1][0]]
            elif (np.where(tempboard == -Randnum, 12, tempboard)>0).all():
                tempboard = np.where(tempboard == -Randnum, 12, tempboard)
                return [np.where(tempboard == tempboard.min())[0][0], np.where(tempboard == tempboard.min())[1][0]], [np.where(tempboard == tempboard.min())[0][0], np.where(tempboard == tempboard.min())[1][0]]
            else:
                min = np.where(tempboard < 0, 12, tempboard)
                max = np.where(tempboard > 0, -12, tempboard)
                return [np.where(min == min.min())[0][0], np.where(min == min.min())[1][0]], [np.where(max == max.max())[0][0], np.where(max == max.max())[1][0]]

    def Move(self, position, direction):
        x = position[0]
        y = position[1]
        if self.board[x][y] > 0:
            if direction == 0:
                self.board[x][y + 1] = self.board[x][y]
                self.board[x][y] = 0.
            elif direction == 1:
                self.board[x + 1][y] = self.board[x][y]
                self.board[x][y] = 0.
            elif direction == 2:
                self.board[x + 1][y + 1] = self.board[x][y]
                self.board[x][y] = 0.
        elif self.board[x][y] < 0:
            if direction == 0:
                self.board[x][y - 1] = self.board[x][y]
                self.board[x][y] = 0.
            elif direction == 1:
                self.board[x - 1][y] = self.board[x][y]
                self.board[x][y] = 0.
            elif direction == 2:
                self.board[x - 1][y - 1] = self.board[x][y]
                self.board[x][y] = 0.

    def GetWinner(self):
        if self.board[0][0] < 0 or (self.board <= 0).all():
            Winner = -1
        elif self.board[4][4] > 0 or (self.board >= 0).all():
            Winner = 1
        else:
            Winner = 0

        return Winner

    def AIMove(self, Position0, Position1):
        Move0 = self.TFMove(Position0)
        Move1 = self.TFMove(Position1)
        if self.Score(Position0, Move0, Position1, Move1) == 0:
            moveDirection = Move0
            Position = Position0
        else:
            moveDirection = Move1
            Position = Position1
        
        return moveDirection, Position

        
    def Saver(self):

        pass

    def show(self):

        pass

    def ChangeBoard(self):
        TempBoard = self.board.copy()
        TempBoard = np.where(TempBoard < 0, -TempBoard + 6, TempBoard)

        return TempBoard

    def TFMove(self, position):
        Board = self.ChangeBoard()
        Board[position[0]][position[1]] = -Board[position[0]][position[1]]
        board_ready = np.reshape(Board,(
            1,
            a_forward.BOARD_SIZE,
            a_forward.BOARD_SIZE,
            a_forward.NUM_CHANNELS))

        Move = sess.run(preValue, feed_dict={x:board_ready})
    
        return Move

    def Score(self, position0, move0, position1, move1):
        if self.board[position0[0]][position0[1]] > 0:
            if move0 == 0:
                attack0 = Rboardvalue[position0[0]][position0[1] + 1] - self.GetPD(self.board, [position0[0], position0[1] + 1])
            elif move0 == 1:
                attack0 = Rboardvalue[position0[0] + 1][position0[1]] - self.GetPD(self.board, [position0[0] + 1, position0[1]])
            elif move0 == 2:
                attack0 = Rboardvalue[position0[0] + 1][position0[1] + 1] - self.GetPD(self.board, [position0[0] + 1, position0[1] + 1])
            if move1 == 0:
                attack1 = Rboardvalue[position1[0]][position1[1] + 1] - self.GetPD(self.board, [position1[0], position1[1] + 1])
            elif move1 == 1:
                attack1 = Rboardvalue[position1[0] + 1][position1[1]] - self.GetPD(self.board, [position1[0] + 1, position1[1]])
            elif move1 == 2:
                attack1 = Rboardvalue[position1[0] + 1][position1[1] + 1] - self.GetPD(self.board, [position1[0] + 1, position1[1] + 1])
        elif self.board[position0[0]][position0[1]] < 0:
            if move0 == 0:
                attack0 = Bboardvalue[position0[0]][position0[1] - 1] - self.GetPD(self.board, [position0[0], position0[1] - 1])
            elif move0 == 1:
                attack0 = Bboardvalue[position0[0] - 1][position0[1]] - self.GetPD(self.board, [position0[0] - 1, position0[1]])
            elif move0 == 2:
                attack0 = Bboardvalue[position0[0] - 1][position0[1] - 1] - self.GetPD(self.board, [position0[0] - 1, position0[1] - 1])
            if move1 == 0:
                attack1 = Bboardvalue[position1[0]][position1[1] - 1] - self.GetPD(self.board, [position1[0], position1[1] - 1])
            elif move1 == 1:
                attack1 = Bboardvalue[position1[0] - 1][position1[1]] - self.GetPD(self.board, [position1[0] - 1, position1[1]])
            elif move1 == 2:
                attack1 = Bboardvalue[position1[0] - 1][position1[1] - 1] - self.GetPD(self.board, [position1[0] - 1, position1[1] - 1])

        return np.argmax([attack0, attack1])

    def GetPD(self, position, sum = 0):
        if self.board[position[0]][position[1]] > 0:
            for Randnum in [1,2,3,4,5,6]:
                P0, P1 = self.ChooseChess(Randnum)
                if P0 == position or P1 == position:
                    sum += 1 
            PD = sum/6 * max(position)
        else:
            for Randnum in [-1,-2,-3,-4,-5,-6]:
                P0, P1 = self.ChooseChess(Randnum)
                if P0 == position or P1 == position:
                    sum += 1
            PD = sum/6 * max(position)

        return PD



EinStein_Game().main()
