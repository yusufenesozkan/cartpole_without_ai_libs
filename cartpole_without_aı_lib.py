#%% Kütüphaneler
import gym
import numpy as np
import time
from collections import deque

#%% Yapay Sinir Ağı sınıfı
class NeuralNetwork:
    def __init__(self, input_size, l1, l2, output_size):
        #ağırlıkları 0.01 çarparak öğrenimi kolaylaştırmayı planlıyorum
        #L1
        self.weights1 = np.random.randn(input_size, l1)*0.1
        self.bias1 = np.random.randn(l1)*0.1
        #L2
        self.weights2 = np.random.randn(l1, l2)*0.1
        self.bias2 = np.random.randn(l2)*0.1
        #Output(L3)
        self.weights3 = np.random.randn(l2, output_size)*0.1
        self.bias3 = np.random.randn(output_size)*0.1

    def relu(self, x):
        return np.maximum(0, x)


    def predict(self, x):
        h1 = self.relu(np.dot(x, self.weights1) + self.bias1 )
        h2 = self.relu(np.dot(h1, self.weights2) + self.bias2)
        output = np.dot(h2, self.weights3) + self.bias3
        return output

    def copy_from(self, other):
        self.weights1 = other.weights1.copy()
        self.bias1 = other.bias1.copy()
        self.weights2 = other.weights2.copy()
        self.bias2 = other.bias2.copy()
        self.weights3 = other.weights3.copy()
        self.bias3 = other.bias3.copy()

#%% ReplayBuffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


#%% Ağırlık güncelleme fonksiyonu
def update_weights(state ,action, reward, next_state , done,
                   network,  target_network, lr=0.0001 ,gamma=0.99):

    # Forward main
    z1 = np.dot(state, network.weights1)  + network.bias1
    a1 = np.maximum(0 , z1)

    z2 = np.dot(a1, network.weights2) + network.bias2
    a2 = np.maximum(0 , z2)

    z3 = np.dot(a2 , network.weights3) + network.bias3
    q_values = z3

    # Calculate targer q value
    next_q_values = target_network.predict(next_state) 
    max_next_q = np.max(next_q_values)
    target_q = reward if done else reward + gamma * max_next_q

    # error
    error = np.zeros_like(q_values)
    error[action] = (q_values[action] - target_q)

    # Backward 
    dL_dz3 = 2 * error  
    dW3 = np.outer(a2, dL_dz3)
    db3 = dL_dz3

    da2 = np.dot(dL_dz3, network.weights3.T)
    dz2 = da2 * (z2 > 0)
    dW2 = np.outer(a1 , dz2)
    db2 = dz2

    da1 = np.dot(dz2, network.weights2.T)
    dz1 = da1 * (z1 > 0)
    dW1 = np.outer(state , dz1)
    db1 = dz1

    # update  main
    network.weights3 -= lr * dW3
    network.bias3 -= lr * db3.reshape(-1)

    network.weights2 -= lr * dW2
    network.bias2 -= lr * db2.reshape(-1)

    network.weights1 -= lr * dW1
    network.bias1 -= lr * db1.reshape(-1)


#%% action selectino
def take_action_w_epsilon(observation, network):
    if np.random.rand() < epsilon:#Random action
        return np.random.choice([0, 1]) 
    else:#via q
        q_values = network.predict(observation)  
        return np.argmax(q_values) 

def take_action_wout_epsilon(observation, network):

    q_values = network.predict(observation)  
    return np.argmax(q_values) 

#%% ön degişkenler
epsilon = 0.1
input_size = 4
layer1 = 256
layer2 = 256
output_size = 2

#main and q networks 
main_network = NeuralNetwork(input_size, layer1, layer2, output_size)
target_network = NeuralNetwork(input_size, layer1, layer2, output_size)
target_network.copy_from(main_network)

env = gym.make("CartPole-v1", render_mode="human")
buffer = ReplayBuffer(capacity=10000)

episode = 0
batch_size = 32
target_update_count = 100
step_count = 0
#%% Main loop
while True:
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = take_action_w_epsilon(state, main_network)
        main_network.predict(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        """if next_state[2] < 0.79 and  next_state[2] > -0.79:
            terminated = False"""
            
        done = terminated or truncated
        pole_angle = state[2]
        cart_position = state[0]
        # Odul hesaplamasını gelecekteki yenilikler icin farklı kullanmayı tercih ettim
        # Çubuğun dikliğine göre ödül
        angle_reward = 1.0 - abs(pole_angle)
        angle_reward = angle_reward ** 2
        max_position = 4.7
        # Arabacığın merkeze yakınlığına göre ödül
        position_reward = 1.0 - abs(cart_position / max_position)  # normalize et
        position_reward = position_reward ** 2
        
        # İkisini birleştir
        reward = (angle_reward + position_reward)/2
        buffer.add(state, action, reward, next_state, done)
        buffer.buffer[0]

        step_count += 1
        if (step_count % target_update_count) == 0:
            target_network.copy_from(main_network)
           
     #-------------------------------------------
        if len(buffer) >= batch_size:
            batch = buffer.get_sample(batch_size)
            for s, a, r, ns, d in zip(*batch):
                update_weights(s, a, r, ns, d, main_network, target_network)
                

        state = next_state
        total_reward += reward
        time.sleep(0.0)

    episode += 1
    epsilon*=0.995

    print(f" Epizod {episode} | Ödül: {int(total_reward)}")
    
    
    
    
    
env.close()

