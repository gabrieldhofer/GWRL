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
        #obstacle = np.random.randint(min(self.rows,self.cols), size=(2,2))
        for i in range(self.rows//4,3*self.rows//4):
            for j in range(self.cols//4,3*self.cols//4):
                self.grid[i,j] = -1*1e8

    def make_prefix_sums(self):
        """
          find path from start to end
        """
        self.grid2=self.grid
        for y in range(1,self.cols):
            self.grid2[0,y]+=self.grid2[0,y-1]
        for x in range(1,self.rows):
            self.grid2[x,0]+=self.grid2[x-1,0]
        for x in range(1,self.rows):
            for y in range(1,self.cols):
                self.grid2[x,y] += \
                        max(self.grid2[x-1,y],self.grid2[x,y-1])


    def find_path(self):
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

    def show_path(self):
        print(self.path) ; print()

    def draw_path(self):
        for point in self.path:
            self.grid[point[0],point[1]]=0

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
def main():
    obj = GWRL(16,16,[ 0.25, 0.25, 0.25, 0.25 ],-1,1)
    obj.generate_obstacle()
    obj.show_array()

    for i in range(100):
        obj.train()
        obj.show_array()
    
    obj.show_array()

    obj.make_prefix_sums()
    obj.find_path()

    obj.draw_path()
    obj.show_heatmap()
    #obj.show_path()


main()


