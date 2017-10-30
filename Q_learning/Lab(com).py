import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):   # argmax인데 숫자가 같으면 랜덤으로 뽑으라는 뜻
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]   # 똑같은 놈들 한곳에 담고
    return pr.choice(indices)              # 그중에 하나 골라라

register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name':'4x4',
            'is_slippery':False}
)
env = gym.make('FrozenLake-v3')   # 환경생성

#### 학습 #####
Q = np.zeros([env.observation_space.n, env.action_space.n])   # 매트릭스 만들기, 2차원 배열, observation은 '상태'임
num_episodes = 2000

# 얼마나 잘했는지 결과를 세이브
rList = []
for i in range(num_episodes):
    # 환경리셋, 처음 상태를 가져온다.
    state = env.reset()
    rAll = 0
    done = False

    # Q-Table 학습 알고리즘
    while not done:
        action = rargmax(Q[state, :])

        # 환경으로부터 보상을 통해 새로운 상태를 가져온다.
        new_state, reward, done, _ = env.step(action)

        # Q-Table 업데이트, 러닝레이트 사용하여
        Q[state,action] = reward + np.max(Q[new_state,:])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
#plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
plt.show()
