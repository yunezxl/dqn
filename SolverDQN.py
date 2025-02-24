from Env_Train import Re_Environment
from GenerateData import GenerationData
import random
from collections import deque
from keras import losses
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Adamax
import tensorflow as tf
import numpy as np
import os


class DQN():
    def __init__(self, CaseParams, Data, Demand_Train, DQNHyparams, DisType):
        self.p1 = CaseParams[0]
        self.p2 = CaseParams[1]
        self.s1 = CaseParams[2]
        self.l1 = CaseParams[3]
        self.s2 = CaseParams[4]
        self.l2 = CaseParams[5]
        self.c = CaseParams[6]

        self.Data = Data
        self.demand_train = Demand_Train
        self.DisType = DisType

        self.state_size = DQNHyparams[0]
        self.action_size = DQNHyparams[1]
        self.gamma = DQNHyparams[2]
        self.epsilon_decay = DQNHyparams[3]
        self.epsilon_min = DQNHyparams[4]
        self.learning_rate = DQNHyparams[5]
        self.batch_size = DQNHyparams[6]
        self.update = DQNHyparams[7]
        self.max_order = self.action_size - 1
        self.env = Re_Environment(self.state_size, self.Data, self.demand_train, self.max_order, None, None, self.c, self.p1, self.s1,
                                  self.l1, self.p2, self.s2, self.l2, len(self.demand_train[0]))

        self.counter = 0
        self.epsilon = 1.0
        self.memory = deque(maxlen=20000)

        self.model = self.build_model()
        self.target_model = self.build_model()

    def build_model(self):
        model = Sequential()

        model.add(Dense(64, input_shape=(self.state_size,), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))

        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def replay(self):

        minibatch = random.sample(self.memory, self.batch_size)
        current_states = np.array([experience[0] for experience in minibatch])
        current_qs_list = np.zeros((self.batch_size, 1, self.env.max_order + 1))
        for k in range(self.batch_size):
            current_qs_list[k] = self.model.predict(current_states[k])
        new_states = np.array([experience[3] for experience in minibatch])
        future_qs_list = np.zeros((self.batch_size, 1, self.env.max_order + 1))
        for k in range(self.batch_size):
            future_qs_list[k] = self.target_model.predict(new_states[k])

        x = []
        y = []

        for i, (current_state, action, reward, next_state, done) in enumerate(minibatch):
            if not done:
                max_fut_q = np.max(future_qs_list[i])
                new_q = reward + self.gamma * max_fut_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[0][int(action)] = new_q
            x.append(current_state[0])
            y.append(current_qs[0])


        self.model.fit(np.array(x), np.array(y), batch_size=self.batch_size, verbose=0, shuffle=False)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


        if self.counter % self.update == 0:
            self.update_target_model()

    def update_target_model(self):

        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
        # print('***** Target network updated *****')

    def train(self):
        done = False
        score = []
        state, _ = self.env.reset(WithTeacher=False)

        while not done:
            state = np.reshape(state, [1, self.state_size])
            action = self.act(state)
            next_state, reward, done, _ = self.env.step(action, WithTeacher=False)
            score.append(reward)
            next_state = np.reshape(next_state, [1, self.state_size])
            self.remember(state, action, reward, next_state, done)
            state = next_state
            self.counter += 1
            if len(self.memory) % self.batch_size == 0:
                self.replay()

        DQN_name = "DQN_Agent_" + str(len(self.Data[0])) + '_' + self.DisType + '_' + 'C=' + str(self.c) + '.h5'
        DQN_folder = 'DQN_Agents'
        Table = "Table2"
        DQN_path = os.path.join(os.getcwd(), Table, DQN_folder, DQN_name)
        self.model.save(DQN_path)

    def save(self, name):
        self.model.save_weights(name)

    def load(self, name):
        self.model.load_weights(name)