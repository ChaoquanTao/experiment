import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def gen_mc_mae():
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 23,
             }
    data = pd.read_csv('data/mcplus2.csv', header=-1)

    df = pd.DataFrame(data)
    print(df)

    xaxis2=np.linspace(0,1,10)

    plt.plot(xaxis2, df[0:1].values.tolist()[0],  label='Total sampling rate 0.1', marker='o', ls='-')
    plt.xlabel('Aerial-ground sample number ratio',font2)
    plt.ylabel('MAE',font2)
    plt.show()

    plt.plot(xaxis2, df[1:2].values.tolist()[0],  label='Total sampling rate 0.2', marker='s', ls='-')
    plt.xlabel('Aerial-ground sample number ratio',font2)
    plt.ylabel('MAE',font2)
    plt.show()

    plt.plot(xaxis2, df[2:3].values.tolist()[0],  label='total sampling rate 0.3', marker='*', ls='-')
    plt.xlabel('Aerial-ground sample number ratio',font2)
    plt.ylabel('MAE',font2)
    plt.show()

    plt.plot(xaxis2, df[3:4].values.tolist()[0],  label='Total sampling rate 0.4', marker='h', ls='-')
    plt.xlabel('Aerial-ground sample number ratio',font2)
    plt.ylabel('MAE',font2)
    plt.show()

    plt.plot(xaxis2, df[4:5].values.tolist()[0],  label='Total sampling rate 0.5', marker='D', ls='-')
    plt.xlabel('Aerial-ground sample number ratio',font2)
    plt.ylabel('MAE',font2)
    plt.show()


#这个函数里的数据是我提前生成保存的，由于比较少就没有用文件保存
def gen_origin_mc():
    x=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    y=[2.7094989133692,1.2365881911218137,0.3648198501218571,0.08848162189650448,
       0.0379965991078555,0.01371392392284477,0.007349389094968703,0.00402315076193938,0.0011628688507640474]
    plt.plot(x,y)
    plt.xlabel('Sampling rate')
    plt.ylabel('MAE')
    plt.show()

gen_mc_mae()
# gen_origin_mc()