import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_random_loss():
    data = pd.read_csv('data/cnnloss.csv', header=-1)

    df = pd.DataFrame(data)
    df=df.loc[:,1:]

    xaxis2=[400,600,800,1000,1200,1400,1600,1800,2000]

    plt.plot(xaxis2, df[0:1].values.tolist()[0], color='r', label='Train with missing rate 0.5', marker='o', ls='-.')
    plt.plot(xaxis2, df[1:2].values.tolist()[0], c='blue', label='Train with missing rate 0.6', marker='s', ls=':')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='orange', label='Train with missing rate 0.7', marker='*', ls='--')
    plt.plot(xaxis2, df[3:4].values.tolist()[0], c='black', label='Train with missing rate 0.8', marker='h', ls='-.')
    plt.plot(xaxis2, df[4:5].values.tolist()[0], c='g', label='Train with missing rate 0.9', marker='D', ls='-.')
    plt.legend()

    plt.xlabel('Iteration step')
    plt.ylabel('Loss')
    plt.show()

def gen_block_loss():
    data = pd.read_csv('data/blockloss.csv', header=-1)

    df = pd.DataFrame(data)
    df=df.loc[:,1:]

    xaxis2=[400,600,800,1000,1200,1400,1600,1800,2000]

    plt.plot(xaxis2, df[0:1].values.tolist()[0], color='r', label='train with missing block 15×15', marker='o', ls='-.')
    plt.plot(xaxis2, df[1:2].values.tolist()[0], c='blue', label='train with missing block 20×20', marker='s', ls=':')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='orange', label='train with missing block 25×25', marker='*', ls='--')
    plt.plot(xaxis2, df[3:4].values.tolist()[0], c='black', label='train with missing block 30×30', marker='h', ls='-.')
    plt.plot(xaxis2, df[4:5].values.tolist()[0], c='g', label='train with missing block 35×35', marker='D', ls='-.')
    plt.legend()

    plt.xlabel('missing rate')
    plt.ylabel('loss')
    plt.show()
# gen_block_loss()
gen_random_loss()