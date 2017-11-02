import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

env = gym.make('FrozenLake-v0')   # 환경생성, 미끄러운 환경 디폴트로 만들어져 있음!! ★★★★★★


# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Discount Factor
learning_rate = .85   # learning_rate, ★★★★★★
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):   # 2000번 에피소드
    # Reset environment and get first new observation, 즉 한번하고 변수 초기화
    state = env.reset()
    rAll = 0
    done = False

    # # The Q-Table learning algorithm - e_greedy
    # e = 1. / ((i//100)+1)   # 숫자가 점점 작아짐
    # while not done:
    #     if np.random.rand(1) < e:   # 랜덤확률이 점점 작아짐
    #         action = env.action_space.sample()   # 액션중에 아무거나 하나 골라라
    #     else:
    #         action = np.argmax(Q[state,:])

    ##################################################################

    # The Q-Table learning algorithm - decaying_add_random_noise
    while not done:
        # Choose an acrion by greedily (with noise) picking from Q Table
        # action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n) / (0.01*(i+1)))   # 노이즈를 추가하는 방법(노이즈 뿌리고 행동 결정), # -1에서 1사이의 수(정규분포)를 1행 4열 리스트로 반환하라, 노이즈:25~0.05
        # action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n) / (0.1*(i+1)))   # 노이즈를 추가하는 방법(노이즈 뿌리고 행동 결정), # -1에서 1사이의 수(정규분포)를 1행 4열 리스트로 반환하라, 노이즈:2.5~005
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n) / (1*(i+1)))   # 노이즈를 추가하는 방법(노이즈 뿌리고 행동 결정), # -1에서 1사이의 수(정규분포)를 1행 4열 리스트로 반환하라, 노이즈:0.25~0.0005

        # Get new state and reward from environment
        new_state,reward,done,_ = env.step(action)   # 행동실행해서 나오는 변수 할당

        # Update Q-Table with new knowledge using decay rate

        Q[state,action] = (1-learning_rate) * Q[state,action] + learning_rate * (reward + dis * np.max(Q[new_state,:]))    #  ★★★★★★★★

        rAll += reward   # 이긴판인지 진판인지 보려고
        state = new_state

    rList.append(rAll)   # 이긴판 진판 싹 모아서 확률 내기 위한 리스트
print("Success rate: " + str(sum(rList)/num_episodes))
print("Final Q-Table Value")
print(Q)
plt.bar(range(len(rList)), rList, color="blue")   # 축, 컨텐트, 색깔
plt.show()