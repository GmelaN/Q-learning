import numpy as np
import matplotlib.pyplot as plt

"""액션 정의: 직영점 간 하루 몇 대의 차를 옮길 것인가?"""
actions = np.arange(-5, 5 + 1)


"""상태 함수"""
# A 직영점과 B 직영점의 모든 가능한 주차된 차 수의 조합
states = []
for a_cars in range(20 + 1):
    for b_cars in range(20 + 1):
        states.append([a_cars, b_cars])

"""가치 함수, 각 직영점이 가질 수 있는 경우의 수는 21가지"""
#현재 시점의 가치
value = np.zeros((20 + 1, 20 + 1))
# 다음 시점의 가치
new_value = np.zeros((20 + 1, 20 + 1))

"""정책 초기화"""
policy = np.zeros((20 + 1, 20 + 1))

"""메모이제이션을 적용한 포아송 분포 메서드 정의"""
p_dict = {}
poisson_upper_bound = 8 + 1

def poisson_distribution(x, lamb):
    global p_dict

    key = (x, lamb)
    if key not in p_dict.keys():
        p_dict[key] = lamb ** x * np.exp(-lamb)/np.math.factorial(x)

    return p_dict[key]


"""가치 함수 업데이트"""
# gamma는 할인율
def calculate_nextV_function(state, action, state_value, gamma = 0.9):
    global poisson_upper_bound

    # 현재 취한 행동으로 인한 보상 감소를 미리 대입해 둠, 보상은 차량 이동량마다 -2씩 부여
    returns = -2 * np.abs(action)

    # 
    A_cars = int(max(min(state[0] - action, 20), 0))
    B_cars = int(max(min(state[1] + action, 20), 0))

    for rentA in range(poisson_upper_bound):
        for rentB in range(poisson_upper_bound):
            for returnA in range(poisson_upper_bound):
                for returnB in range(poisson_upper_bound):
                    rent_prob = poisson_distribution(rentA, 3)
                    rent_prob *= poisson_distribution(rentB, 4)

                    Alot_rent_fin = min(A_cars, rentA)
                    Blot_rent_fin = min(B_cars, rentB)

                    reward = (Alot_rent_fin + Blot_rent_fin) * 10

                    return_prob = poisson_distribution(returnA, 3)
                    return_prob *= poisson_distribution(returnB, 2)

                    next_A_cars = int(max(min(A_cars - Alot_rent_fin + returnA, 20), 0))
                    next_B_cars = int(max(min(B_cars - Blot_rent_fin + returnB, 20), 0))

                    total_prob = rent_prob * return_prob

                    returns += total_prob * (reward + gamma * state_value[next_A_cars, next_B_cars])

    return returns

