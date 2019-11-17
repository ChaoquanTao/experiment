
import pandas as pd
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt
import heapq

#读数据,得到dataframe
def read_data():
    data_path = "../data/data/DlRsrpSinrStats.txt"

    # data_set = np.loadtxt(data_path,delimiter='	',skiprows=1);
    data_set = pd.read_table(data_path,delimiter='\t')
    # print(data_set.shape)

    df = pd.DataFrame(data_set)
    df.loc[:,'rsrp'] = 10*np.log10(df.loc[:,'rsrp']) + 30
    # print(df)
    return df

# 返回该时隙空地数据
def get_time_slot_data(n):
    df = read_data()
    ts_data = (df.loc[3200*n:(n+1)*3200-1].sort_values(by='IMSI'))['rsrp']
    # plt.imshow(np.array(ts_data[0:1600]).reshape(40, 40))
    # plt.show()
    # print((ts_data.shape)[1])
    return np.array(ts_data)

def gen_mask(m, n, prob_masked=0.5):

    return 1 - bernoulli.rvs(p=prob_masked, size=(m, n))

# 在某个时隙n的数据里按照随机缺失率选择
def random_select(data,n,mr):
    if data is None:
        data = get_time_slot_data(n)[1600:3200]
    mask = gen_mask(40,40,mr)

    sparse_data = np.array(data).reshape(40,40) * mask
#    plt.imshow(sparse_data)
#    plt.show()
    return sparse_data.reshape(1,1600)
    # plt.hist(sparse_data[sparse_data!=0])
    # plt.show()
    # print(sparse_data)

def select_lastk(array,judge,k):
    idx = np.argsort(judge)
    # print(idx)
    idxs = idx[0:k]
    for i in idxs:
        array[i] = 0
    return array

# 主动选择某个时隙中空中数据的关键点
def active_select(data,n,mr):
    if data is None:
        data = get_time_slot_data(n)[1600:3200]
    # print(data[data==0])

#    plt.imshow(np.array(data).reshape(40,40))
#    plt.show()

    #数值取整
    data2 = np.floor(data)
#    data2 = data.copy()
    # print(data)

    #通过概率的倒数表示信息量
    selfInfo=[]
    # selfInfo = np.array(selfInfo)
    for dt in data2:
        # np.append(selfInfo,len(data)/np.sum(data==dt))
        selfInfo.append(len(data2)/np.sum(data2==dt))

#    print('selfinfo')
#    print(selfInfo)

    data4 = data.copy()
    data4 = select_lastk(data4,selfInfo,int(1600*mr))
 
#    plt.imshow(np.array(data4).reshape(40,40))
#    plt.show()

    # 计算代表性
    sim=[]
    data3 = np.array(data).reshape(40,40)
    for i in range(len(data3)):
        for j in range(len(data3[0])):
            # if data3[i][j]==0:
            #     print(data3[i][j])
            #     continue
            diff = 0
            cnt =0
            if(i-1)>=0:
                if j-1>=0:
                    diff += np.abs(data3[i-1][j-1]-data3[i][j])/np.abs(data3[i][j]) #左上
                    cnt+=1
                diff += np.abs(data3[i-1][j]-data3[i][j])/np.abs(data3[i][j])       #正上
                cnt+=1
                if j+1<len(data3[0]):
                    diff += np.abs(data3[i-1][j+1]-data3[i][j])/np.abs(data3[i][j]) #右上
                    cnt += 1
            if j-1>=0:
                diff += np.abs(data3[i][j-1]-data3[i][j])/np.abs(data3[i][j]) #正左
                cnt += 1
            if j+1 < len(data3[0]):
                diff += np.abs(data3[i][j+1]-data3[i][j])/np.abs(data3[i][j]) #正右
                cnt += 1
            if i+1<len(data3):
                if j-1>=0:
                    diff += np.abs(data3[i+1][j-1]-data3[i][j])/np.abs(data3[i][j]) #左下
                    cnt += 1
                if j+1<len(data3[0]):
                    diff += np.abs(data3[i+1][j+1]-data3[i][j])/np.abs(data3[i][j]) #右下
                    cnt += 1
            # print(diff)
            sim.append(diff/cnt)
    # 排序sim
#    print('sim')
#    print(sim)
    data5 = data.copy()
    data5 = select_lastk(data5,sim,int(1600*mr))
#    plt.imshow(np.array(data5).reshape(40,40))
#    plt.show()

    judge = np.multiply(selfInfo,sim)
#    print('judge')
#    print(judge)

    data6 = data.copy()
    data6 = select_lastk(data6,judge,int(1600*mr))
#    plt.imshow(np.array(data6).reshape(40, 40))
#    plt.show()
    return data6

def rand_active_select(data,n,mr,sr1,sr2):
    if data is None:
        data = get_time_slot_data(n)[1600:3200]

    
    active_res = active_select(data,n,1-sr1)
    random_res = random_select(data,n,1-sr2)
#    print(active_res.shape)
#    print(random_res.shape)
    
    logical_res = np.logical_xor(active_res,random_res)
    res = logical_res*data   
#    print(sum(sum(res!=0)))
##    print('random active selection')
    # plt.imshow(np.array(res).reshape(40,40))
    # plt.show()
    return res
# random_select(0,0.7)

  
# rand_active_select(data=None,n=0,mr = 0.8,sr1=0.95,sr2=0.85)
#active_select(data=None,n=0,mr = 0.7)
    
