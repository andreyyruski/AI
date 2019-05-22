import gym
import random
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

env = gym.make('CartPole-v1')
env.reset()
epochs = 3

def build_model(input_size, output_size):
    model = Sequential()

    model.add(Dense(265, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(output_size, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam())
    
    return model

def train_model(training_data, model = None):
    X = np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]))
    y = np.array([i[1] for i in training_data]).reshape(-1, len(training_data[0][1]))
    
    # build new model if there is no one provided
    if model == None:
        model = build_model(input_size=len(X[0]), output_size=len(y[0]))
        
    model.fit(X, y, epochs=epochs)
    return model

def play(training_data, score_requirement, model = None,
 number_games = 100, max_goal_steps = 500, predict = False,
 render = False, verbose = False):
    accepted_scores = []
    for game_index in range(number_games):
        score = 0
        game_memory = []
        previous_observation = []
        for step_index in range(max_goal_steps):
            if len(previous_observation) == 0:
                action = random.randrange(0, 2)
            elif predict and model is not None:
                #reshape observation
                observ = np.array(previous_observation.reshape(-1, len(training_data[0][0])))
                
                prediction = model.predict(observ)
                action = np.argmax(prediction)
            else:
                action = random.randrange(0, 2)

            observation, reward, done, info = env.step(action)
            if render:
                env.render()
            
            if len(previous_observation) > 0:
                game_memory.append([previous_observation, action])
                
            previous_observation = observation
            score += reward
            if done:
                break
            
        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                training_data.append([data[0], output, score])
        
        env.reset()

        if verbose:
            print('Score = {}'.format(score))

    print(accepted_scores)
    return training_data


def clean_low_train_data(training_data, score_threshold):
    x = len(training_data)
    training_data = [data for data in training_data if data[2] >= score_threshold]
    x -= len(training_data)
    print('Cleand {} events with score lower than {}'.format(x, score_threshold))
    return training_data

trained_model = None

# first train on the random data
training_data = []
training_data = play(training_data, score_requirement = 60, number_games = 10000)

# then train on minimum scores
score_requirement_list = [100,200,500]
for score_requirement in score_requirement_list:
    print('----------------> Picking up score {} and up <----------------'.format(score_requirement))
    # play
    training_data = play(training_data, score_requirement = score_requirement, 
        model = trained_model, max_goal_steps = 500, 
        number_games = 100, predict = True, verbose = True)

    # train    
    trained_model = train_model(training_data)
    
    # clean low score traning data
    #training_data = [training_data[0]]
    training_data = clean_low_train_data(training_data,
        score_requirement)

play(training_data, score_requirement = 500,
    model = trained_model, predict = True, verbose = True)

play(training_data, score_requirement = 500,
    model = trained_model, predict = True, render = True, verbose = True)
