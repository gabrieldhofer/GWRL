import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import math

class GWRL:
    def __init__(self,rows,cols,policy,reward,gamma):
        """ 
            Initialize the grid and member variables
        """
        self.rows = rows
        self.cols = cols
        self.policy = policy
        self.reward = reward
        self.gamma = gamma
        self.grid = np.zeros([rows,cols], dtype=float)

    def update_grid(self,rows_lst,cols_lst):
        """ 
            Iterate through grid 
        """
        temp_grid = np.zeros([self.rows,self.cols], dtype=float)
        for i in rows_lst:
            for j in cols_lst:
                if (not (i==0 and j==0)) and (not (i==self.rows-1 and j==self.cols-1)):
                    temp_grid[ i,j ] += \
                            self.policy[0] * (self.reward + self.gamma * \
                            self.grid[ (i+1 if i<self.rows-1 else i),j ])
                    temp_grid[ i,j ] += \
                            self.policy[0] * (self.reward + self.gamma * \
                            self.grid[ (i-1 if i>0 else i),j ])
                    temp_grid[ i,j ] += \
                            self.policy[0] * (self.reward + self.gamma * \
                            self.grid[ i,(j+1 if j<self.cols-1 else j) ])
                    temp_grid[ i,j ] += \
                            self.policy[0] * (self.reward + self.gamma * \
                            self.grid[ i,(j-1 if j>0 else j) ])
        self.grid = temp_grid

    def train(self):
        """ 
            Update grid values based on the policy  
        """
        self.update_grid(range(self.rows),range(self.cols))
        self.update_grid(range(self.rows)[::-1],range(self.cols))
        self.update_grid(range(self.rows),range(self.cols)[::-1])
        self.update_grid(range(self.rows)[::-1],range(self.cols)[::-1])

    def show_array(self):
        """
            Print the array to the terminal
        """
        for i in range(self.rows):
            for j in range(self.cols):
                print("%.2f" % self.grid[i,j], end='   ')
            print()
        print() ; print()

    def generate_obstacle(self):
        """ 
            Create an obstacle (a square in the middle of the image) 
        """
        for i in range(3*self.rows//8, 6*self.rows//8):
            for j in range(3*self.cols//8, 6*self.cols//8):
                self.grid[i,j] = -1*1e15

    def make_prefix_sums(self):
        """ 
            find path from start to end 
        """
        self.grid=self.grid
        for y in range(1,self.cols):
            self.grid[0,y]+=self.grid[0,y-1]
        for x in range(1,self.rows):
            self.grid[x,0]+=self.grid[x-1,0]
        for x in range(1,self.rows):
            for y in range(1,self.cols):
                self.grid[x,y] += max(self.grid[x-1,y],self.grid[x,y-1])

    def find_path(self):
        """ 
            start at 0,0 and choose the smallest neighboring square 
        """
        x,y=0,0
        self.path=[(x,y)]
        while(x!=self.rows-1 or y!=self.cols-1):
            mx=-1*1e20
            (x2,y2) = (x,y)
            if x+1<self.rows:
                if self.grid[x+1,y] > mx:
                    (x2,y2) = (x+1,y)
                    mx = self.grid[x2,y2]                    
            if y+1<self.cols:
                if self.grid[x,y+1] > mx:
                    (x2,y2) = (x,y+1)
                    mx = self.grid[x2,y2]                    
            (x,y) = (x2,y2)
            self.path.append((x,y)) 

    def draw_path(self):
        """ 
            paints each square in the path the same color 
        """
        for point in self.path: 
            self.grid[point[0],point[1]]=0

    def show_heatmap(self):
        """ 
            display heatmap 
        """
        plt.imshow(self.grid, cmap=plt.cm.bwr) 
        plt.show()

    def fix_it(self):
        """
            chnages the values of cells in the array to clearly
            show the path and the obstacle
        """
        obstacle_color= self.grid[self.rows-1,0]
        for i in range(self.rows):
            for j in range(self.cols):
                if (i,j) not in self.path:
                    self.grid[i,j]=obstacle_color//2
        for i in range(3*self.rows//8, 6*self.rows//8):
            for j in range(3*self.cols//8, 6*self.cols//8):
                if (i,j) not in self.path:
                    self.grid[i,j] = obstacle_color

    def fix_it_no_obstacle(self):
        """
            chnages the values of cells in the array to clearly
            show the path and the obstacle
        """
        obstacle_color= self.grid[self.rows-1,0]
        for i in range(self.rows):
            for j in range(self.cols):
                if (i,j) not in self.path:
                    self.grid[i,j]=obstacle_color

"""
    Main, entry point
"""
import time
def main():
    obj = GWRL(32, 32, [ 0.25, 0.25, 0.25, 0.25 ], -1, 1)
    obj.generate_obstacle()
    for i in range(130): 
        obj.train()
    obj.make_prefix_sums()
    obj.find_path()
    obj.draw_path()
    obj.fix_it()
    #obj.fix_it_no_obstacle()
    obj.show_heatmap()

main()


