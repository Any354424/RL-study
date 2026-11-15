import numpy as np
from The_world import State,move,states,local,gamma,boundary_punish

if __name__ == "__main__":
    policy=np.zeros((9,5),dtype=float)  #写了一个策略矩阵，里面每一行的元素是a1,a2,a3,a4,a5的概率
    v={(s.x,s.y):1  for s in states}    #初始的state value
    times=0 #初始化次数
    while True:
        v_old=v.copy()  #保存上一次结果
        for i in states:    #对于所有的state
            times+=1    #记录迭代次数
            action_value_set=list() #创建临时action value列表，比较里面的元素
            for j in range(1,6):    #5种运动
                action_value=0  #初始化action value
                nex=move(i,j)   #动
                action_value+=nex[2]+gamma *v[(nex[0],nex[1])]  #action value的定义
                action_value_set.append(action_value)  #记录单个state所有的action value
            v[(i.x,i.y)]=max(action_value_set) #贪心

        count = 0
        for xy,value in v.items():
            if max(abs(v[k] - v_old[k]) for k in v)<1e-6:
                count+=1
        if count==9:
            break
    print(v,'\n',v_old,'\n',times)
