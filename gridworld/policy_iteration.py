import numpy as np
from The_world import State,move,states,local,gamma,boundary_punish

# 策略迭代：固定策略做评估，评估完做贪心改进，改不动了就收敛
# 和值迭代的区别：值迭代每次只改进"一步"，策略迭代每次把当前策略算到收敛再改

if __name__ == "__main__":
    #初始化策略：每个状态先随便选一个动作，这里统一选 a5（不动）
    policy = {(s.x, s.y): 5 for s in states}
    v = {(s.x, s.y): 0 for s in states}   # 当前策略下每个状态的价值
    times_1 =0
    times = 0            # 策略迭代次数

    while True:
        # 策略不变，按贝尔曼公式迭代，直到v不再变化
        while True:
            v_old = v.copy()              # 保存上一次结果，用来判断收没收敛
            for i in states:              # 对于所有的 state
                a = policy[(i.x, i.y)]    # 当前策略在这个状态选的动作
                nex = move(i, a)          # 走一步：拿到下一个位置和这一步的奖励
                v[(i.x, i.y)] = nex[2] + gamma * v[(nex[0], nex[1])]
                times_1 += 1
            if max(abs(v[k] - v_old[k]) for k in v) < 1e-6:              # 所有状态都几乎不变，评估收敛
                break

        # 用刚算出来的 v，对每个状态贪心地挑action value最大的动作
        new_policy = {}
        for i in states:
            action_value_set = []
            for j in range(1, 6):         # 遍历运动
                nex = move(i, j)        #动
                action_value = nex[2] + gamma * v[(nex[0], nex[1])]     #
                action_value_set.append(action_value)   #记录action value
            best_a = int(np.argmax(action_value_set)) + 1   #最大值的索引+1
            old_a = policy[(i.x, i.y)]      #修改字典
            # 如果新动作没有比原来更好（并列），就保留原来的动作
            if action_value_set[best_a - 1] - action_value_set[old_a - 1] > 1e-12:
                new_policy[(i.x, i.y)] = best_a
            else:
                new_policy[(i.x, i.y)] = old_a
        if new_policy == policy:    #收敛
            break
        policy = new_policy
        times += 1
    print(v,'\n',v_old,'\n',times_1,'\n',times)