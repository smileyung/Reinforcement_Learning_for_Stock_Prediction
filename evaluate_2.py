import keras
from keras.models import load_model

from agent.agent import Agent
from functions import *
import sys

def myevaluate(arg1, arg2):
    # try:
        #  if len(sys.argv) != 3:
        #     print ("Usage: python evaluate.py [stock] [model]")
        #     exit()

        # stock_name, model_name = sys.argv[1], sys.argv[2]
        
        stock_name, model_name = arg1, arg2
        model = load_model("models/" + model_name)
        window_size = model.layers[0].input.shape.as_list()[1]

        agent = Agent(window_size, True, model_name)
        data = getStockDataVec(stock_name)
        l = len(data) - 1
        batch_size = 32

        state = getState(data, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []

        for t in range(l):
            action = agent.act(state)

            # sit
            next_state = getState(data, t + 1, window_size + 1)
            reward = 0

            if action == 1: # buy
                agent.inventory.append(data[t])
                print ("Buy: " + formatPrice(data[t]))

            elif action == 2 and len(agent.inventory) > 0: # sell
                bought_price = agent.inventory.pop(0)
                reward = max(data[t] - bought_price, 0)
                total_profit += data[t] - bought_price
                print ("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

            done = True if t == l - 1 else False
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state

            if done:
                print ("--------------------------------")
                print (stock_name + " Total Profit: " + formatPrice(total_profit))
                print ("--------------------------------")
                return formatPrice(total_profit)

            if len(agent.memory) > batch_size:
                    agent.expReplay(batch_size) 

    # finally:
        # exit()
        
if __name__ == '__main__':
    
    import os
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    
    #print('Call it locally')
    #cool_func()
    arg1 = '^GSPC_2011'
    
    mylist = []
    for i in range(0,1010,10):
        arg2 = 'model_ep'+str(i)
        mylist.append(myevaluate(arg1, arg2))
        
    import pickle
    with open("model_ep_test.txt", "wb") as fp:   #Pickling
        pickle.dump(mylist, fp)