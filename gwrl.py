import numpy as np
import matplotlib.pyplot as plt


class GWRL:
    def __init__(self,rows,cols,policy,reward,gamma):
        """ Initialize the grid """
        self.rows = rows
        self.cols = cols
        self.policy = policy
        self.reward = reward
        self.gamma = gamma
        self.grid = np.zeros([rows,cols], dtype=float)

    def train(self):
        """ Update grid values based on the policy  """
        temp_grid = np.zeros([self.rows,self.cols], dtype=float)
        for i in range(self.rows):
            for j in range(self.cols):
                if (not (i==0 and j==0)) and (not (i==self.rows-1 and j==self.cols-1)):
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ (i+1 if i<self.rows-1 else i),j ])
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ (i-1 if i>0 else i),j ])
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ i,(j+1 if j<self.cols-1 else j) ])
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ i,(j-1 if j>0 else j) ])
        self.grid = temp_grid

    def generate_obstacle(self):
        """ add an obstacle """
        for i in range(3*self.rows//8, 6*self.rows//8):
            for j in range(3*self.cols//8, 6*self.cols//8):
                self.grid[i,j] = -1*1e6

    def make_prefix_sums(self):
        """
          find path from start to end
        """
        self.grid_sum=self.grid
        for y in range(1,self.cols):
            self.grid_sum[0,y]+=self.grid_sum[0,y-1]
        for x in range(1,self.rows):
            self.grid_sum[x,0]+=self.grid_sum[x-1,0]
        for x in range(1,self.rows):
            for y in range(1,self.cols):
                self.grid_sum[x,y] += \
                        max(self.grid_sum[x-1,y],self.grid_sum[x,y-1])

    def find_path(self):
        x,y=0,0
        self.path=[(x,y)]

        while(x!=self.rows-1 or y!=self.cols-1):
            mx=-1*1e20
            (x2,y2) = (x,y)
            if x+1<self.rows:
                if self.grid_sum[x+1,y] > mx:
                    (x2,y2) = (x+1,y)
                    mx = self.grid_sum[x2,y2]                    
            if y+1<self.cols:
                if self.grid_sum[x,y+1] > mx:
                    (x2,y2) = (x,y+1)
                    mx = self.grid_sum[x2,y2]                    
            (x,y) = (x2,y2)
            self.path.append((x,y)) 
        

    def show_path(self):
        print(self.path) ; print()

    def draw_path(self):
        for point in self.path:
            self.grid[point[0],point[1]]=0

    def show_path_and_obstacle(self):
        self.output = np.zeros([self.rows, self.cols], dtype=float)
        for point in self.path:
            self.output[point[0],point[1]]=0
        for i in range(self.rows//4, 3*self.rows//4):
            for j in range(self.cols//4, 3*self.cols//4):
                self.output[i,j] = -1*1e6
        plt.imshow(self.output, cmap=plt.cm.bwr)
        plt.show()

    def show_array(self):
        for i in range(self.rows):
            for j in range(self.cols):
                print("%.2f" % self.grid[i,j], end='   ')
            print()
        print() ; print()

    def show_heatmap(self):
        plt.imshow(self.grid, cmap=plt.cm.bwr)
        plt.show()


"""

"""
import time
def main():
    obj = GWRL(8, 8, [ 0.25, 0.25, 0.25, 0.25 ], -1, 1)
    obj.generate_obstacle()
    obj.show_array()

    for i in range(10):
        obj.train()
        obj.show_array()
    
    print('here 1')
    obj.show_array()

    obj.make_prefix_sums()
    print('here 2')
    obj.find_path()
    print('here 3')
    obj.show_path()

    obj.draw_path()
    #obj.show_heatmap()


    obj.show_path_and_obstacle()

main()


