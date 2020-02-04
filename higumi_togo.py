#coding:utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import os
import a_forward
import a_backward
import MCTS
import math
import random

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
saver.restore(sess, ckpt.model_checkpoint_path)

class EinStein_Game():
    def __init__(self):
        self.board = np.zeros([5,5])
        self.PI = []
        self.CI = []
        self.RCSS = []  #红色棋子初始设定
        self.BCSS = []
        self.mode = ['game', 'test'][1]
        self.AImode = ['MCTS', 'UCT', 'TF', 'MCTS&TF', 'UCT&TF', 'Doge'][2]
        self.Player = 0
        self.UCTSTEPS = 3000
        self.MCTSSTEPS = 3000

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
                    while True:
                        os.system('cls')
                        print(self.board)
                        if self.Player > 0:
                            print('#####红方时间#####')
                        else:
                            print('*****蓝方时间*****')
                        rand = int(input('请输入随机数,输入0以悔棋：'))
                        if rand == 0:
                            if len(self.PI) == 1:
                                print('爪巴！赶紧的')
                                continue
                            else:
                                self.Player = -self.Player
                                del self.PI[len(self.PI) - 1]
                                del self.CI[len(self.CI) - 1]
                                self.board = np.array(self.PI[len(self.PI) - 1]).reshape([5,5])
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
                        self.Move(position, moveDirection, rand)
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
                        while True:
                            os.system('cls')
                            print(self.board)
                            if self.Player > 0:
                                print('#####红方时间#####')
                            else:
                                print('*****蓝方时间*****')
                            rand = int(input('请输入随机数,输入0以悔棋：'))
                            if rand == 0:
                                if len(self.PI) == 1:
                                    print('爪巴！赶紧的')
                                    continue
                                else:
                                    self.Player = -self.Player
                                    del self.PI[len(self.PI) - 1]
                                    del self.CI[len(self.CI) - 1]
                                    self.board = np.array(self.PI[len(self.PI) - 1]).reshape([5,5])
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
                            self.Move(position, moveDirection, rand)
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
                        while True:
                            os.system('cls')
                            print(self.board)
                            if self.Player > 0:
                                print('#####红方时间#####')
                            else:
                                print('*****蓝方时间*****')
                            rand = int(input('请输入随机数,输入0以悔棋：'))
                            if rand == 0:
                                if len(self.PI) == 1:
                                    print('爪巴！赶紧的')
                                    continue
                                else:
                                    self.Player = -self.Player
                                    del self.PI[len(self.PI) - 1]
                                    del self.CI[len(self.CI) - 1]
                                    self.board = np.array(self.PI[len(self.PI) - 1]).reshape([5,5])
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
                            self.Move(position, moveDirection, rand)
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
                if len(self.CI) == 0:
                    print('还未进行棋局！！')
                    _ = input('回车以继续：')
                    continue
                self.Save()
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
        self.PI = []
        self.CI = []
        self.RCSS = ['R:']
        self.BCSS = ['B:']
        self.board = np.zeros([5,5])
        self.board[0][0] =  red[0]
        self.RCSS.append('A5-')
        self.RCSS.append(str(red[0]))
        self.RCSS.append(';')
        self.board[0][1] =  red[1]
        self.RCSS.append('B5-')
        self.RCSS.append(str(red[1]))
        self.RCSS.append(';')
        self.board[0][2] =  red[2]
        self.RCSS.append('C5-')
        self.RCSS.append(str(red[2]))
        self.RCSS.append(';')
        self.board[1][0] =  red[3]
        self.RCSS.append('A4-')
        self.RCSS.append(str(red[3]))
        self.RCSS.append(';')
        self.board[1][1] =  red[4]
        self.RCSS.append('B4-')
        self.RCSS.append(str(red[4]))
        self.RCSS.append(';')
        self.board[2][0] =  red[5]
        self.RCSS.append('A3-')
        self.RCSS.append(str(red[5]))
        self.RCSS.append(';')
        self.board[2][4] = blue[0]
        self.BCSS.append('E3-')
        self.BCSS.append(str(blue[0]))
        self.BCSS.append(';')
        self.board[3][3] = blue[1]
        self.BCSS.append('D2-')
        self.BCSS.append(str(blue[1]))
        self.BCSS.append(';')
        self.board[3][4] = blue[2]
        self.BCSS.append('E2-')
        self.BCSS.append(str(blue[2]))
        self.BCSS.append(';')
        self.board[4][2] = blue[3]
        self.BCSS.append('C1-')
        self.BCSS.append(str(blue[3]))
        self.BCSS.append(';')
        self.board[4][3] = blue[4]
        self.BCSS.append('D1-')
        self.BCSS.append(str(blue[4]))
        self.BCSS.append(';')
        self.board[4][4] = blue[5]
        self.BCSS.append('E1-')
        self.BCSS.append(str(blue[5]))
        self.BCSS.append(';')
        self.PI.append(list(self.board.reshape(25)))
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

    def Move(self, position, direction, rand):
        x = position[0]
        y = position[1]
        if self.board[x][y] > 0:
            if direction == 0:
                self.board[x][y + 1] = self.board[x][y]
                self.board[x][y] = 0.
                y += 1
            elif direction == 1:
                self.board[x + 1][y] = self.board[x][y]
                self.board[x][y] = 0.
                x += 1
            elif direction == 2:
                self.board[x + 1][y + 1] = self.board[x][y]
                self.board[x][y] = 0.
                x += 1 
                y += 1
        elif self.board[x][y] < 0:
            if direction == 0:
                self.board[x][y - 1] = self.board[x][y]
                self.board[x][y] = 0.
                y -= 1
            elif direction == 1:
                self.board[x - 1][y] = self.board[x][y]
                self.board[x][y] = 0.
                x -= 1
            elif direction == 2:
                self.board[x - 1][y - 1] = self.board[x][y]
                self.board[x][y] = 0.
                x -= 1
                y -= 1

        self.appendCI(x, y, rand)

    def GetWinner(self):
        if self.board[0][0] < 0 or (self.board <= 0).all():
            Winner = -1
        elif self.board[4][4] > 0 or (self.board >= 0).all():
            Winner = 1
        else:
            Winner = 0

        return Winner

    def AIMove(self, Position0, Position1):
        
        if Position0 == Position1:
            moveDirection = self.GetAIMove(Position0)
            Position = Position0
        else:
            Move0 = self.GetAIMove(Position0)
            Move1 = self.GetAIMove(Position1)
            if self.Score(Position0, Move0, Position1, Move1) == 0:
                moveDirection = Move0
                Position = Position0
            else:
                moveDirection = Move1
                Position = Position1
        
        return Position, moveDirection

        
    def Save(self):
        Team1 = input('请输入队伍1名称：')
        Team2 = input('请输入队伍2名称：')
        Location = input('请输入比赛地点：')
        Name = input('请输入竞赛名称：')

        FileName = Team1 + 'vs' + Team2 + '-' + time.strftime("%Y%m%d%H%M%S", time.localtime()) + '.txt'
        Text1 = '#[' + Team1 + '][' + Team2 + '][' + time.strftime("%Y.%m.%d %H:%M:%S", time.localtime()) + ' ' + Location + '][' + Name + '];'
        File = open(FileName, 'w')
        File.write(Text1)
        File.write('\r')
        for RC in self.RCSS:
            File.write(RC)
        File.write('\r')
        for BC in self.BCSS:
            File.write(BC)
        File.write('\r')
        for Step in range(len(self.CI)):
            File.write(str(Step + 1))
            File.write(self.CI[Step])
            File.write('\r')
        File.close()

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
            a_forward.NUM_CHANNELS)) / 12.

        Move = sess.run(preValue, feed_dict={x:board_ready})
    
        return Move[0]

    def Score(self, position0, move0, position1, move1):
        if self.board[position0[0]][position0[1]] > 0:
            if move0 == 0:
                attack0 = Rboardvalue[position0[0]][position0[1] + 1] - self.GetPD([position0[0], position0[1] + 1])
            elif move0 == 1:
                attack0 = Rboardvalue[position0[0] + 1][position0[1]] - self.GetPD([position0[0] + 1, position0[1]])
            elif move0 == 2:
                attack0 = Rboardvalue[position0[0] + 1][position0[1] + 1] - self.GetPD([position0[0] + 1, position0[1] + 1])
            if move1 == 0:
                attack1 = Rboardvalue[position1[0]][position1[1] + 1] - self.GetPD([position1[0], position1[1] + 1])
            elif move1 == 1:
                attack1 = Rboardvalue[position1[0] + 1][position1[1]] - self.GetPD([position1[0] + 1, position1[1]])
            elif move1 == 2:
                attack1 = Rboardvalue[position1[0] + 1][position1[1] + 1] - self.GetPD([position1[0] + 1, position1[1] + 1])
        elif self.board[position0[0]][position0[1]] < 0:
            if move0 == 0:
                attack0 = Bboardvalue[position0[0]][position0[1] - 1] - self.GetPD([position0[0], position0[1] - 1])
            elif move0 == 1:
                attack0 = Bboardvalue[position0[0] - 1][position0[1]] - self.GetPD([position0[0] - 1, position0[1]])
            elif move0 == 2:
                attack0 = Bboardvalue[position0[0] - 1][position0[1] - 1] - self.GetPD([position0[0] - 1, position0[1] - 1])
            if move1 == 0:
                attack1 = Bboardvalue[position1[0]][position1[1] - 1] - self.GetPD([position1[0], position1[1] - 1])
            elif move1 == 1:
                attack1 = Bboardvalue[position1[0] - 1][position1[1]] - self.GetPD([position1[0] - 1, position1[1]])
            elif move1 == 2:
                attack1 = Bboardvalue[position1[0] - 1][position1[1] - 1] - self.GetPD([position1[0] - 1, position1[1] - 1])

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

    def appendCI(self, x, y, rand):
        chess = self.board[x][y]
        if chess > 0:
            chess = 'R' + str(chess)
        else:
            chess = 'B' + str(-chess)
        x = str(int(5 - x))
        if y == 0:
            y = 'A'
        elif y == 1:
            y = 'B'
        elif y == 2:
            y = 'C'
        elif y == 3:
            y = 'D'
        elif y == 4:
            y = 'E'
        self.CI.append(':' + str(abs(rand)) + ';(' + chess + ',' + y + x + ')')

    def GetAIMove(self, position):
        if self.board[position[0]][position[1]] > 0:
            if position[0] == 4:
                Move = 0
            elif position[1] == 4:
                Move = 1
            else:
                if self.AImode == 'TF':
                    Move = self.TFMove(position)
                elif self.AImode == 'UCT&TF':
                    Move = self.UCTMove(position)
                elif self.AImode == 'UCT':
                    Move = self.UCTMove(position, False)
                elif self.AImode == 'Doge':
                    Move = random.randint(0, 2)
                elif self.AImode == 'MCTS':
                    WinRate0 = MCTS.MCTS(self.board.copy(), position, 0, STEPS = self.MCTSSTEPS, is_TF = False)
                    WinRate1 = MCTS.MCTS(self.board.copy(), position, 1, STEPS = self.MCTSSTEPS, is_TF = False)
                    WinRate2 = MCTS.MCTS(self.board.copy(), position, 2, STEPS = self.MCTSSTEPS, is_TF = False)
                    Move = np.argmax([WinRate0, WinRate1, WinRate2])
                elif self.AImode == 'MCTS&TF':
                    WinRate0 = MCTS.MCTS(self.board.copy(), position, 0, STEPS = self.MCTSSTEPS)
                    WinRate1 = MCTS.MCTS(self.board.copy(), position, 1, STEPS = self.MCTSSTEPS)
                    WinRate2 = MCTS.MCTS(self.board.copy(), position, 2, STEPS = self.MCTSSTEPS)
                    Move = np.argmax([WinRate0, WinRate1, WinRate2])
        elif self.board[position[0]][position[1]] < 0:
            if position[0] == 0:
                Move = 0
            elif position[1] == 0:
                Move = 1
            else:
                if self.AImode == 'TF':
                    Move = self.TFMove(position)
                elif self.AImode == 'UCT&TF':
                    Move = self.UCTMove(position)
                elif self.AImode == 'UCT':
                    Move = self.UCTMove(position, False)
                elif self.AImode == 'Doge':
                    Move = random.randint(0, 2)
                elif self.AImode == 'MCTS':
                    WinRate0 = MCTS.MCTS(self.board.copy(), position, 0, STEPS = self.MCTSSTEPS, is_TF = False)
                    WinRate1 = MCTS.MCTS(self.board.copy(), position, 1, STEPS = self.MCTSSTEPS, is_TF = False)
                    WinRate2 = MCTS.MCTS(self.board.copy(), position, 2, STEPS = self.MCTSSTEPS, is_TF = False)
                    Move = np.argmax([WinRate0, WinRate1, WinRate2])
                elif self.AImode == 'MCTS&TF':
                    WinRate0 = MCTS.MCTS(self.board.copy(), position, 0, STEPS = self.MCTSSTEPS)
                    WinRate1 = MCTS.MCTS(self.board.copy(), position, 1, STEPS = self.MCTSSTEPS)
                    WinRate2 = MCTS.MCTS(self.board.copy(), position, 2, STEPS = self.MCTSSTEPS)
                    Move = np.argmax([WinRate0, WinRate1, WinRate2])
        
        return Move

    def UCTMove(self, position, is_TF = True):
        N = [0, 0, 0]
        UCTvalue = [0, 0, 0]
        WinSum = [0, 0, 0]
        for Step in range(self.UCTSTEPS):
            board = self.board.copy()
            i = np.argmax(UCTvalue)
            WinSum[i] += MCTS.MCTS(board, position, i, STEPS = 1, is_TF = is_TF)
            N[i] += 1
            UCTvalue[0] = WinSum[0] / (N[0] + 1e-99) + (2 * math.log(Step + 1)/(N[0] + 1e-99)) ** 0.5
            UCTvalue[1] = WinSum[1] / (N[1] + 1e-99) + (2 * math.log(Step + 1)/(N[1] + 1e-99)) ** 0.5
            UCTvalue[2] = WinSum[2] / (N[2] + 1e-99) + (2 * math.log(Step + 1)/(N[2] + 1e-99)) ** 0.5
        #print('WinSum:', WinSum)
        #print('     N:', N)

        return np.argmax(np.array(WinSum) / np.array(N))

if __name__ == '__main__':
    EinStein_Game().main()
