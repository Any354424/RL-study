'''网格示意图
s1 s2 s3
s4 s5 s6
s7x s8 s9*
'''

# 1向上 2向右 3向下 4向左 5不动

#模型规则
boundary_punish = -1 #边界惩罚
gamma = 0.9   #衰减率

#定义state
class State:
    def __init__(self,index,x,y,inter_reward):  #传入编号，xy坐标，进入格子的奖励
        self.index = index
        self.x = x
        self.y = y
        self.inter_reward=inter_reward

#定义网格
states = [
        State('s1', 1, 3, 0),
        State('s2', 2, 3, 0),
        State('s3', 3, 3, 0),
        State('s4', 1, 2, 0),
        State('s5', 2, 2, 0),
        State('s6', 3, 2, 0),
        State('s7', 1, 1, -1),
        State('s8', 2, 1, 0),
        State('s9', 3, 1, 1),
        ]

#添加检索
local = {}
for s in states:
    local[(s.x, s.y)] = s

#定义一个action
def move(s,action): #输入当前位置和动作，返回x,y值和获得的奖励
    x=s.x
    y=s.y
    reward=0
    if action == 1:
        if y == 3:
            reward += boundary_punish
        else:
            y += 1
            reward += local[(x, y)].inter_reward
    if action == 2:
        if x == 3:
            reward += boundary_punish
        else:
            x += 1
            reward += local[(x, y)].inter_reward
    if action == 3:
        if y == 1:
            reward += boundary_punish
        else:
            y += -1
            reward += local[(x, y)].inter_reward
    if action == 4:
        if x == 1:
            reward+=boundary_punish
        else:
            x += -1
            reward += local[(x, y)].inter_reward
    if action == 5:
        reward += s.inter_reward
    return (x,y,reward)