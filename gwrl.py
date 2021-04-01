import numpy as np
import matplotlib.pyplot as plt


class GWRL:
    def __init__(self,rows,cols,policy,reward,gamma):
        """
            Our policy equally weights moving
            up, down, left, and right
        """
        self.rows = rows
        self.cols = cols
        self.policy = policy
        self.reward = reward
        self.gamma = gamma
        self.grid = np.zeros([rows,cols], dtype=float)

    def train(self):
        """
            Update grid values based on the policy  
        """
        temp_grid = np.zeros([self.rows,self.cols], dtype=float)
        for i in range(self.rows):
            for j in range(self.cols):
                if (not (i==0 and j==0)) and (not (i==self.rows-1 and j==self.cols-1)):
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ (i+1 if i<self.rows-1 else i),j ])
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ (i-1 if i>0 else i),j ])
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ i,(j+1 if j<self.cols-1 else j) ])
                    temp_grid[ i,j ] += self.policy[0] * (self.reward + self.gamma * self.grid[ i,(j-1 if j>0 else j) ])

        self.grid = temp_grid
    
    def generate_obstacles(self):
        """
            create some obstacles randomly
        """
        pass

    def show(self):
        print(self.grid)



                
obj = GWRL(4,4,[ 0.25, 0.25, 0.25, 0.25 ],-1,1)
for i in range(10):
    obj.train()
    obj.show()
    print()


