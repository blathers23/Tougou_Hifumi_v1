#coding:utf-8

import numpy as np
import pandas as pd
import os

class EinStein_Game():
    def __init__(self):
        self.board = np.zeros([5,5])
        self.PI = []


    def main(self):
        while True:
            print('###############')
            print('##Higumi_Togo##')
            print('###############')
            print()
            Tip1 = input('请输入"N"开始新游戏：')
            if Tip1 == "N":
                Tip2 = input('人人对弈请输入"H", 人机对弈请输入"A":')
                if Tip2 == "H":
                    redBoard = input('请输入红方(左上方)棋子布局,以逗号间隔：')
                    redBoard = [float(n) for n in redBoard.split(',')]
                    blueBoard = input('请输入蓝方（右下方）棋子布局,以逗号间隔：')
                    blueBoard = [-float(n) for n in blueBoard.split(',')]
                    Player = input('请输入先手方：')
                    Player = float(Player)
                    self.createBoard(redBoard, blueBoard)
                    while True:
                        os.system('cls')
                        print(self.board)
                        if Player > 0:
                            print('#####红方时间#####')
                        else:
                            print('*****蓝方时间*****')
                        rand = int(input('请输入随机数：'))
                        if Player * rand < 0:
                            print('!!!!!WARNING!!!!!')
                        chessPosition1, chessPosition2 = self.ChooseChess(rand)
                        Chess = int(input('请输入移动的棋子：'))
                        if Chess not in [self.board[chessPosition1[0], chessPosition1[1]], self.board[chessPosition2[0], chessPosition2[1]]]:
                            print('!!!!!WARNING!!!!!')
                            while True:
                                Chess = int(input('请输入移动的棋子：'))
                                if Chess in [self.board[chessPosition1[0], chessPosition1[1]], self.board[chessPosition2[0], chessPosition2[1]]]:
                                    break
                                else:
                                    print('!!!!!WARNING!!!!!')
                        if Chess == self.board[chessPosition1[0], chessPosition1[1]]:
                            position = chessPosition1
                        else:
                            position = chessPosition2
                        moveDirection = self.GetMove(position)
                        self.Move(position, moveDirection)
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
                        Player = -Player
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


        pass

    def GetWinner(self):
        if self.board[0][0] < 0 or (self.board <= 0).all():
            Winner = -1
        elif self.board[4][4] > 0 or (self.board >= 0).all():
            Winner = 1
        else:
            Winner = 0

        return Winner

    def AI(self):

        pass

    def show(self):

        pass

    def regret(self):

        pass


EinStein_Game().main()
