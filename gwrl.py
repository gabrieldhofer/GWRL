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
        temp = np.zeros([self.rows,self.cols], dtype=float)
        for i in range(self.rows):
            for j in range(self.cols):
                if (not (i==0 and j==0)) and (not (i==self.rows-1 and j==self.cols-1)):
                    temp[i,j] += self.policy[0] * (self.reward + \
                        self.gamma * self.grid[ (i+1 if i<self.rows-1 else i),j ])
                    temp[i,j] += self.policy[1] * (self.reward + \
                        self.gamma * self.grid[ (i-1 if i>0 else i),j ])
                    temp[i,j] += self.policy[2] * (self.reward + \
                        self.gamma * self.grid[ i,(j+1 if j<self.cols-1 else j) ])
                    temp[i,j] += self.policy[3] * (self.reward + \
                        self.gamma * self.grid[ i,(j-1 if j>0 else j) ])
        self.grid = temp
    
    def generate_obstacle(self):
        """ add an obstacle """
        obstacle = np.random.randint(min(self.rows,self.cols), size=(2,2))
        for i in range(obstacle[0,:]):
          for j in range(obstacle[1,:]):
            self.grid[i,j] = -50 ## <-- the obstacle value

    def get_path(self):
        """
          find path from start to end
        """
        x,y=0,0
        loc=(0,0)
        self.path=[loc]
        while(x!=self.rows-1 and y!=self.cols-1):
          mx=-1*1e8
          if(x-1>=0):
            mx = self.grid[x-1,y] if self.gird[x-1,y]>mx else mx
            loc=(x-1,y)
          if(x+1<self.rows):
            mx = self.grid[x+1,y] if self.gird[x+1,y]>mx else mx
            loc=(x+1,y)
          if(y-1>=self.cols):
            mx = self.grid[x,y-1] if self.gird[x,y-1]>mx else mx
            loc=(x,y-1)
          if(y+1<self.cols):
            mx = self.grid[x,y+1] if self.gird[x,y+1]>mx else mx
            loc=(x,y+1)
          self.path.append(loc) 
              
    def show_array(self):
        """ print the grid """
        print(self.grid)

    def show_heatmap(self):
      pass



"""
  
"""
obj = GWRL(4,4,[ 0.25, 0.25, 0.25, 0.25 ],-1,1)
for i in range(10):
    obj.train()
    obj.show()
    print()


