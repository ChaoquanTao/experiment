import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_cnn_mae():
    data = pd.read_csv('data/blockmae.csv', header=-1)

    df = pd.DataFrame(data)
    print(df)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['', 'missing block 15×15', 'missing block 20×20', 'missing block 25×25', 'missing block 30×30']
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    xaxis2=[10,15,20,25,30]

    plt.plot(xaxis2, df[0:1].values.tolist()[0], color='r', label='train with missing block 10×10', marker='o', ls='-.')
    plt.plot(xaxis2, df[1:2].values.tolist()[0], c='blue', label='train with missing block 15×15', marker='s', ls=':')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='orange', label='train with missing block 20×20', marker='*', ls='--')
    plt.plot(xaxis2, df[3:4].values.tolist()[0], c='black', label='train with missing block 25×25', marker='h', ls='-.')
    plt.plot(xaxis2, df[4:5].values.tolist()[0], c='g', label='train with missing block 30×30', marker='D', ls='-.')
    plt.legend()
    plt.xticks(np.arange(10,30,5))
    plt.xlabel('width of missing block')
    plt.ylabel('mae')
    plt.show()

def gen_other_mae():
    data = pd.read_csv('data/block_loc_loc.csv', header=-1)

    df = pd.DataFrame(data)
    print(df)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['', 'missing block 15×15', 'missing block 20×20', 'missing block 25×25', 'missing block 30×30']
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    xaxis2=[10,15,20,25,30]

    data2 = pd.read_csv('data/blockmae.csv', header=-1)

    df2 = pd.DataFrame(data2)

    xaxis2 = [10, 15, 20, 25, 30]
    plt.plot(xaxis2, df2[0:1].values.tolist()[0], color='r', label='our scheme', marker='o', ls='-.')


    plt.plot(xaxis2, df[1:2].values.tolist()[0], c='blue', label='KNN', marker='s', ls=':')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='orange', label='MC', marker='*', ls='--')
    plt.plot(xaxis2, df[3:4].values.tolist()[0], c='black', label='MICE', marker='h', ls='-.')

    plt.legend()
    plt.xticks(np.arange(10,30,5))
    plt.xlabel('width of missing block')
    plt.ylabel('mae')
    plt.show()
gen_other_mae()
