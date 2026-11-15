import random
from The_world import State,move,states,local,gamma,boundary_punish

#计算一个return
if __name__ == "__main__":
    return1=0
    state_now=states[0]
    for i in range(10):
        action=random.randint(1, 5)
        print(f'第{i}步：从{state_now.index}采取{action}，',end=' ')
        re=move(state_now,action)
        return1+=re[2] * gamma ** i
        state_now=local.get((re[0],re[1]))
        print(f'移动到{state_now.index},return为{return1}')
