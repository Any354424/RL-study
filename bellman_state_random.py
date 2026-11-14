import random
'''网格示意图
s1 s2 s3
s4 s5 s6
s7x s8 s9*
'''

#模型规则
boundary_punish = -1
gamma=0.9

#定义state
class state:
    def __init__(self,index,x,y,inter_reward):
        self.index = index
        self.x = x
        self.y = y
        self.inter_reward=inter_reward
states=[
state('s1',1,3,0),
state('s2',2,3,0),
state('s3',3,3,0),
state('s4',1,2,0),
state('s5',2,2,0),
state('s6',3,2,0),
state('s7',1,1,-1),
state('s8',2,1,0),
state('s9',3,1,1),
]

local={}
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

#计算一个return
return1=0
state_now=states[0]

for i in range(10):
    action=random.randint(1, 5)
    print(f'第{i}步：从{state_now.index}采取{action}，',end=' ')
    re=move(state_now,action)
    return1+=re[2] * gamma ** i
    state_now=local.get((re[0],re[1]))
    print(f'移动到{state_now.index},return为{return1}')
