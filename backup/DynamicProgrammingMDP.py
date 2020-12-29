"""
    Part 1
"""
S = [i for i in range(16)]  # 状态空间
A = ['n', 'e', 's', 'w']  # 动作空间
# P, R 动态生成

actions = {'n': -4, 'e': 1, 's': 4, 'w': -1}


def dynamics(s, a):
    """
        Args:
            s state 0-15
            a action ['n', 'e', 's', 'w']
        Returns:
            tuple(s_prime, reward, is_end)

        0/15 terminal
    """
    s_prime = s
    if (s % 4 == 0 and a == "w") or (s < 4 and a == "n") or (
            (s + 1) % 4 == 0 and a == "e") or (s > 11
                                               and a == "s") or s in [0, 15]:
        pass
    else:
        ds = actions[a]
        s_prime = s + ds
    reward = 0 if s in [0, 15] else -1
    is_end = True if s in [0, 15] else False
    return s_prime, reward, is_end


def P(s, a, s1):
    # 状态转移函数 状态s执行a动作转移到s1的概率 此处只有1或者0
    s_prime, _, _ = dynamics(s, a)
    if s_prime == s1:
        return 1
    else:
        return 0


def R(s, a):
    _, r, _ = dynamics(s, a)
    return r


gamma = 1
# MDP 5 tuple
MDP = S, A, R, P, gamma

"""
    Part 2
"""


# 完全随机策略
def uniform_pi(MDP=MDP, V=None, s=None, a=None):
    _, A, _, _, _ = MDP
    n = len(A)
    return 0 if n == 0 else 1.0 / n


# 完全贪婪策略
def greedy_pi(MDP, V, s, a):
    # 当多个状态具有相同最大值的情况，贪婪策略将从多个具有最大值行为中随机选择一个
    S, A, P, R, gamma = MDP
    max_v, a_max_v = -float('inf'), []
    for a_opt in A:
        # 后续状态的最大价值以及到达该状态的行为
        s_prime, reward, _ = dynamics(s, a_opt)
        v_s_prime = get_value(V, s_prime)

        if v_s_prime > max_v:
            max_v = v_s_prime
            a_max_v = [a_opt]
        elif v_s_prime == max_v:
            a_max_v.append(a_opt)

    n = len(a_max_v)
    if n == 0:
        return 0.0
    return 1.0 / n if a in a_max_v else 0.0


"""
    Part 3
"""


def get_reward(R, s, a):
    # 获取奖励值
    return R(s, a)


def display_V(V):
    for i in range(16):
        print('{0:>6.1f}'.format(V[i]), end=" ")
        if (i + 1) % 4 == 0:
            print("")
    print()


def set_value(V, s, v):
    # set value dict
    V[s] = v


def get_value(V, s):
    # get value
    return V[s]


def get_Pi(Pi, s, a, MDP=None, V=None):
    return Pi(MDP, V, s, a)


def get_prob(P, s, a, s1):
    # 获取状态转移概率
    return P(s, a, s1)


def compute_v(MDP, V, Pi, s):
    """
        给定MDP，依据某一策略Pi和当前状态价值函数V，计算某状态s的价值
        get_Pi: 基于当前策略π，状态为s的情况下选择动作a的概率
    """
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_Pi(Pi, s, a, MDP, V) * compute_q(MDP, V, s, a)
    return v_s


def compute_q(MDP, V, s, a):
    """
        根据给定的MDP， 价值函数V， 计算状态行为对Q(s, a)
        get_prob: 获取状态转移概率 在该环境中，状态s使用动作a到达状态s'的概率为1或者0
        get_value: 获取价值字典，即获取V[s]
        get_reward: 获取在状态s使用动作a得到的奖励值

        对应贝尔曼方程中的括号中部分
    """
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
    q_sa = get_reward(R, s, a) + gamma * q_sa

    return q_sa


def update_V(MDP, V, Pi):
    """
        根据给定的MDP 和 Pi，更新当前的V
    """
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    v_list = list()
    for s in S:
        v = compute_v(MDP, V_prime, Pi, s)
        v_list.append(v)
    k = 0
    for s in S:
        set_value(V, s, v_list[k])
        k += 1
    return V


"""
    Part 4
"""
"""
    策略评估
    分别评估在随机策略和贪婪策略下16个状态的最终价值
"""


def policy_evaluate(MDP, V, pi, n):
    """
        param:
            MDP 建立的MDP模型
            V 初始状态值函数
            pi 使用的策略
            n 迭代次数
        return：
            V_eva 在n次迭代之后按照给定策略MDP的V值分布
    """
    for i in range(n):
        V = update_V(MDP, V, pi)
    return V


'''
V = [0 for _ in range(16)]  # 初始化价值
V_eva = policy_evaluate(MDP, V, uniform_pi, 100)
display_V(V_eva)
'''

'''V = [0 for _ in range(16)]  # 初始化价值
V_eva = policy_evaluate(MDP, V, greedy_pi, 100)
display_V(V_eva)'''

"""
    策略迭代
    每迭代1次改善1次策略
"""


def policy_iterate(MDP, V, Pi, n, m):
    for i in range(m):
        V = policy_evaluate(MDP, V, Pi, n)
        Pi = greedy_pi
    return V


'''V = [0 for _ in range(16)]  # 初始化价值
V_eva = policy_iterate(MDP, V, greedy_pi, 1, 100)
display_V(V_eva)'''


"""
    价值迭代
"""


def compute_v_from_max_q(MDP, V, s):
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qs = compute_q(MDP, V, s, a)
        if qs >= v_s:
            v_s = qs
    return v_s


def update_V_without_pi(MDP, V):
    """
    无策略，仅依靠状态价值来更新状态价值

    """
    S, _, _, _, _ = MDP
    V_p = V.copy()
    for s in S:
        compu_v = compute_v_from_max_q(MDP, V_p, s)  # 通过max q更新v
        set_value(V_p, s, compu_v)
    return V_p


def value_iterate(MDP, V, n):
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V


V = [0 for _ in range(16)]  # 初始化价值
v_s = value_iterate(MDP, V, 4)
display_V(v_s)
