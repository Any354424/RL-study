import random
from The_world import State,move,states,local,gamma,boundary_punish
'''网格示意图
s1 s2 s3
s4 s5 s6
s7x s8 s9*
'''
if __name__ == "__main__":
    #定义return和出发
    return_now=0
    best_return=float('-inf')
    best_path=''
    state_now=states[0]
    path_this_time = '没有找到s9'
    for i in range(1000):
        return_now=0
        state_now = states[0]
        path_this_time = 's1->'
         #开始记录这次尝试的路径
        for j in range(1000):
            action=random.randint(1, 5)  #选择一个行动
            re = move(state_now,action)  #记录行动结果
            return_now += re[2] * gamma ** j  #更新state value
            state_now=local.get((re[0],re[1]))   #更新state
            if state_now.index == 's9':   #更新记录
                path_this_time += f'{str(state_now.index)}'
                break
            else:
                path_this_time += f'{str(state_now.index)}->'
        if return_now > best_return:
            best_return = return_now
            best_path = path_this_time


    print(f's0的最大state value={best_return},路径是{best_path}')