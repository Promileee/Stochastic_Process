import random
import math
import numpy as np
import matplotlib.pyplot as plt

"""任务1:蒙特卡洛仿真"""

# --- 1. 定义参数 ---
lambda_4_3 = 0.1
lambda_4_2 = 0.05
lambda_4_1 = 0.02
lambda_3_2 = 0.15
lambda_3_1 = 0.08
lambda_2_1 = 0.2

num_products = 20000  # 仿真产品数量，越多越精确
max_sim_time = 50.0  # 最大仿真时间
time_step = 0.5  # 统计概率的时间步长

# 所有可能的状态
states = [1, 2, 3, 4]

# 存储每个产品在每个统计时间点的状态
# 例如：product_history[product_idx] = [(time1, state1), (time2, state2), ...]
product_histories = []


# --- 2. 模拟单次产品退化路径 ---
def simulate_single_product():
    current_state = 4
    current_time = 0.0
    history = []  # 记录 (时间, 状态) 对
    history.append((current_time, current_state))

    while current_state != 1 and current_time < max_sim_time:
        if current_state == 4:
            rates = {'3': lambda_4_3, '2': lambda_4_2, '1': lambda_4_1}
        elif current_state == 3:
            rates = {'2': lambda_3_2, '1': lambda_3_1}
        elif current_state == 2:
            rates = {'1': lambda_2_1}
        else:  # current_state == 1 (absorbed state)
            break

        total_rate_out = sum(rates.values())

        # 停留时间
        delta_t_stay = -1 / total_rate_out * math.log(random.random())
        current_time += delta_t_stay

        # 确定下一个状态
        if current_time >= max_sim_time:  # 模拟时间已到
            break

        r = random.random()
        cumulative_prob = 0
        for target_state_str, rate in rates.items():
            prob_transition = rate / total_rate_out
            cumulative_prob += prob_transition
            if r <= cumulative_prob:
                next_state = int(target_state_str)
                break

        current_state = next_state
        history.append((current_time, current_state))

    return history


# --- 3. 运行蒙特卡洛仿真 ---
for _ in range(num_products):
    product_histories.append(simulate_single_product())

# --- 4. 统计结果 ---
time_points = np.arange(0, max_sim_time + time_step, time_step)
state_probabilities = np.zeros((len(time_points), len(states) + 1))  # +1 是因为状态从1开始

for i, t_check in enumerate(time_points):
    state_counts_at_t = {s: 0 for s in states}
    for history in product_histories:
        # 找到在 t_check 时刻产品所处的状态
        # 遍历历史记录，找到在 t_check 之前的最后一个状态
        current_state_at_t = 4  # 初始状态
        for time_h, state_h in history:
            if time_h <= t_check:
                current_state_at_t = state_h
            else:
                break  # 已经超过当前检查点
        state_counts_at_t[current_state_at_t] += 1

    for state_id in states:
        state_probabilities[i, state_id] = state_counts_at_t[state_id] / num_products

# --- 5. 可视化 ---
plt.figure(figsize=(10, 6))
for state_id in states:
    plt.plot(time_points, state_probabilities[:, state_id], label=f'P({state_id})(t)')

plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.title('Monte Carlo Simulation of Product Degradation')
plt.legend()
plt.grid(True)
plt.show()

"""任务2:微分方程结果"""
# --- 6. 初始化路径 ---
diff_probabilities_shape=(len(states), len(time_points))
diff_probabilities = np.zeros(diff_probabilities_shape)

# --- 7. 计算路径 ---
for i in range(len(time_points)):
    time_point = time_points[i]
    for j in range(len(states)):
        if j == 0:
            diff_probabilities[j,i] = 1 - (38/3)*math.exp(-0.17*time_point) - (20/3)*math.exp(-0.23*time_point) + (55/3)*math.exp(-0.2*time_point)
        elif j == 1:
            diff_probabilities[j,i] = 10*math.exp(-0.17*time_point) + (25/3)*math.exp(-0.23*time_point) - (55/3)*math.exp(-0.2*time_point)
        elif j == 2:
            diff_probabilities[j,i] = (5/3)*(math.exp(-0.17*time_point) - math.exp(-0.23*time_point))
        else:
            diff_probabilities[j,i] = math.exp(-0.17*time_point)

# --- 8. 可视化 ---
plt.figure(figsize=(10, 6))
for state_id in states:
    plt.plot(time_points, diff_probabilities[state_id - 1, :], label=f'P({state_id})(t)')

plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.title('Difference Equation Calculation of Product Degradation')
plt.legend()
plt.grid(True)
plt.show()

# --- 9. 对比 ---
# 对比蒙特卡洛仿真结果与微分方程计算结果
plt.figure(figsize=(12, 10))

# 创建2x2的子图布局
plt.subplot(2, 2, 1)  # 状态1
plt.plot(time_points, diff_probabilities[0, :], 'r-', linewidth=2, label='Diff Eq')
plt.plot(time_points, state_probabilities[:, 1], 'b--', linewidth=1.5, label='Monte Carlo')
plt.title('State 1 Probability')
plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)  # 状态2
plt.plot(time_points, diff_probabilities[1, :], 'r-', linewidth=2, label='Diff Eq')
plt.plot(time_points, state_probabilities[:, 2], 'b--', linewidth=1.5, label='Monte Carlo')
plt.title('State 2 Probability')
plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)  # 状态3
plt.plot(time_points, diff_probabilities[2, :], 'r-', linewidth=2, label='Diff Eq')
plt.plot(time_points, state_probabilities[:, 3], 'b--', linewidth=1.5, label='Monte Carlo')
plt.title('State 3 Probability')
plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)  # 状态4
plt.plot(time_points, diff_probabilities[3, :], 'r-', linewidth=2, label='Diff Eq')
plt.plot(time_points, state_probabilities[:, 4], 'b--', linewidth=1.5, label='Monte Carlo')
plt.title('State 4 Probability')
plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.suptitle('Comparison of Monte Carlo Simulation vs Differential Equation Solution', fontsize=14)
plt.show()

"""任务3：考虑维修的蒙特卡洛仿真"""

# --- 10. 更新部分参数 ---
mu_1_4 = 0.05

num_products = 1000000  # 仿真产品数量，越多越精确

# 存储每个产品在每个统计时间点的状态
product_histories_fix = []

# --- 11. 模拟单次产品退化路径 ---
def simulate_single_product_fix():
    current_state = 4
    current_time = 0.0
    history = []  # 记录 (时间, 状态) 对
    history.append((current_time, current_state))

    while current_time < max_sim_time:
        if current_state == 4:
            rates = {'3': lambda_4_3, '2': lambda_4_2, '1': lambda_4_1}
        elif current_state == 3:
            rates = {'2': lambda_3_2, '1': lambda_3_1}
        elif current_state == 2:
            rates = {'1': lambda_2_1}
        else:  # current_state == 1 (absorbed state)
            rates = {'4': mu_1_4}

        total_rate_out = sum(rates.values())

        # 停留时间
        delta_t_stay = -1 / total_rate_out * math.log(random.random())
        current_time += delta_t_stay

        # 确定下一个状态
        if current_time >= max_sim_time:  # 模拟时间已到
            break

        r = random.random()
        cumulative_prob = 0
        next_state = current_state  # 默认不转移，如果所有概率都很小
        for target_state_str, rate in rates.items():
            prob_transition = rate / total_rate_out
            cumulative_prob += prob_transition
            if r <= cumulative_prob:
                next_state = int(target_state_str)
                break

        current_state = next_state
        history.append((current_time, current_state))

    return history


# --- 12. 运行蒙特卡洛仿真 ---
for _ in range(num_products):
    product_histories_fix.append(simulate_single_product_fix())

# --- 13. 统计结果 ---
state_probabilities_fix = np.zeros((len(time_points), len(states) + 1))  # +1 是因为状态从1开始

for i, t_check in enumerate(time_points):
    state_counts_at_t_fix = {s: 0 for s in states}
    for history in product_histories_fix:
        # 找到在 t_check 时刻产品所处的状态
        # 遍历历史记录，找到在 t_check 之前的最后一个状态
        current_state_at_t = 4  # 初始状态
        for time_h, state_h in history:
            if time_h <= t_check:
                current_state_at_t = state_h
            else:
                break  # 已经超过当前检查点
        state_counts_at_t_fix[current_state_at_t] += 1

    for state_id in states:
        state_probabilities_fix[i, state_id] = state_counts_at_t_fix[state_id] / num_products

# --- 14. 可视化 ---
plt.figure(figsize=(10, 6))
for state_id in states:
    plt.plot(time_points, state_probabilities_fix[:, state_id], label=f'P({state_id})(t)')

plt.xlabel('Time (t)')
plt.ylabel('Probability')
plt.title('Monte Carlo Simulation of Product Degradation Considering Fix')
plt.legend()
plt.grid(True)
plt.show()