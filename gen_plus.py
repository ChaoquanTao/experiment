import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_mae(filename,idx):
    data = pd.read_csv(filename,header=-1)
    df = pd.DataFrame(data)
    # print(df)

    bs_40 = df[9*idx:(idx+1)*9]
    print()
    xaxis=bs_40.iloc[:,1]
    print(xaxis)
    mae1=bs_40.iloc[:,2]
    mae2 = bs_40.iloc[:, 3]
    mae3 = bs_40.iloc[:, 4]
    mae4 = bs_40.iloc[:, 5]
    mae5 = bs_40.iloc[:, 6]
    return xaxis,mae1,mae2,mae3,mae4,mae5

xaxis,mae1,mae2,mae3,mae4,mae5 = get_mae('./data/plusmae.csv',3)
plt.plot(xaxis,mae1,color='b',label='missing rate 0.5',marker='v',ls='-.')
plt.plot(xaxis,mae2,color='g',label='missing rate 0.6',marker='s',ls='--')
plt.plot(xaxis,mae3,color='r',label='missing rate 0.7',marker='D',ls=':')
plt.plot(xaxis,mae4,color='c',label='missing rate 0.8',marker='o',ls='-.')
plt.plot(xaxis,mae5,color='k',label='missing rate 0.9',marker='x',ls='-.')
plt.xlabel('heights of UAVs')
plt.ylabel('mae')
plt.legend()
plt.show()