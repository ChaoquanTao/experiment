import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def gen_five():
    # name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
    # num_list = [1.5, 0.6, 7.8, 6]
    # num_list1 = [1, 2, 3, 1]
    x = list(range(5))
    total_width, n = 0.5, 5
    width = total_width / n

    # plt.bar(x, num_list, width=width, label='boy', fc='y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
    # plt.legend()
    # plt.show()

    mae=[]

    data = pd.read_csv('data/cnnmae2.csv',header=-1)

    df = pd.DataFrame(data)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs=['r','y','b','g','orange']
    labels=['missing rate 0.5','missing rate 0.6','missing rate 0.7','missing rate 0.8','missing rate 0.9']
    ticks=[0.5,0.6,0.7,0.8,0.9]
    # print(tmodel)
    for i in range(5):
        dt = df[i*2][::2]
        print(dt)
        if i!=0:
            for j in range(len(x)):
                x[j] = x[j] + width

        plt.bar(x, dt, width=width, label=labels[i], fc=fcs[i])
        # break

    # my_x_ticks = np.arange(-5, 5, 0.5)
    my_y_ticks = np.arange(0, 0.06, 0.01)
    # plt.xticks(my_x_ticks)
    # plt.tick_params(labels)
    plt.xticks(range(5),ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('missing rate when training model')
    plt.ylabel('mae')
    plt.legend()
    plt.show()

#用多少缺失率训练，就用多少缺失率测试
def gen_random2as_line():
    x = list(range(5))
    x2 = x.copy()
    x3 = x.copy()
    total_width, n = 1, 5
    width = total_width / n

    for i in range(len(x)):
        x2[i] = x[i] + width
        x3[i] = x[i] + 2 * width
    mae = []

    origin_files = ['data/random_mae_origin_pathloss.csv', 'data/data/active_mae_origin_pathloss.csv',
                    'data/data/random_active_mae_origin_pathloss.csv']
    log_files = ['data/random_mae_log_pathloss.csv', 'data/data/active_mae_log_pathloss.csv',
                 'data/data/random_active_mae_log_pathloss.csv']
    random_data = pd.read_csv(log_files[0], header=-1)
    active_data = pd.read_csv(log_files[1], header=-1)
    active_random_data = pd.read_csv(log_files[2], header=-1)
    df1 = pd.DataFrame(random_data)
    df2 = pd.DataFrame(active_data)
    df3 = pd.DataFrame(active_random_data)
    print(df1.shape)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['random ', 'active', 'random_active']
    # ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    ticks=[0.5,0.55,0.6,0.65,0.70,0.75,0.80,0.85,0.9,0.95]
    randline=[]
    activeline=[]
    rand_active_line=[]
    # print(df1[5])
    for i in range(10):
        # print(df1.iat[5+(i*2)][i*2])
        randline.append(df1.iat[5+i,i])
        activeline.append(df2.iat[5+i,i])
        rand_active_line.append(df3.iat[5+i,i])
    plt.plot(ticks,randline,c='r')
    plt.plot(ticks, activeline, c='y')
    plt.plot(ticks, rand_active_line, c='b')
    plt.show()

#横轴训练缺失率和test缺失率，纵轴mae
def gen_as_train2test():
    x = list(range(5))
    x2 = x.copy()
    x3 = x.copy()
    total_width, n = 1, 5
    width = total_width / n

    for i in range(len(x)):
        x2[i] = x[i] + width
        x3[i] = x[i] + 2 * width
    mae = []

    log_files = ['data/random_mae_log_pathloss.csv', 'data/data/active_mae_log_pathloss.csv',
                 'data/data/random_active_mae_log_pathloss.csv']
    random_data = pd.read_csv(log_files[0], header=-1)
    active_data = pd.read_csv(log_files[1], header=-1)
    active_random_data = pd.read_csv(log_files[2], header=-1)
    df1 = pd.DataFrame(random_data)
    df2 = pd.DataFrame(active_data)
    df3 = pd.DataFrame(active_random_data)
    print(df1.shape)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['random ', 'active', 'random_active']
    # ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    ticks = [0.5, 0.55, 0.6, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95]

    test_mrs=[4,6,8]
    for i in test_mrs:
        randline = []
        activeline = []
        rand_active_line = []
        # print(df1.iat[5+(i*2)][i*2])
        randline = df1[5:15][i]
        activeline=df2[5:15][i]
        rand_active_line=df3[5:15][i]
        print(randline)
        plt.plot(ticks, randline, c='r')
        plt.plot(ticks, activeline, c='y')
        plt.plot(ticks, rand_active_line, c='b')
        plt.show()



#生成不同缺失率交叉对比图
def gen_random2as():
    x = list(range(5))
    x2=x.copy()
    x3=x.copy()
    total_width, n = 1, 5
    width = total_width / n

    for i in range(len(x)):
        x2[i] = x[i] + width
        x3[i] = x[i] + 2*width
    mae=[]

    origin_files=['data/random_mae_origin_pathloss.csv','data/data/active_mae_origin_pathloss.csv','data/data/random_active_mae_origin_pathloss.csv']
    log_files=['data/random_mae_log_pathloss.csv','data/data/active_mae_log_pathloss.csv','data/data/random_active_mae_log_pathloss-fix.csv']
    random_data = pd.read_csv(log_files[0],header=-1)
    active_data = pd.read_csv(log_files[1],header=-1)
    active_random_data = pd.read_csv(log_files[2],header=-1)
    df1 = pd.DataFrame(random_data)
    df2 = pd.DataFrame(active_data)
    df3 = pd.DataFrame(active_random_data)

    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs=['r','y','b','g','orange']
    labels=['random ','active','random_active']
    ticks=[0.5,0.6,0.7,0.8,0.9]
    # print(tmodel)
    for i in range(5):
        dt1 = df1.ix[i:i+1,0:4]
        # print(dt1)
        dt2 = df2.ix[i:i+1,0:4]
        print(dt2.values.tolist())
        dt3 = df3.ix[i:i+1,0:4]
        # if i!=0:
        #     for j in range(len(x)):
        #         x[j] = x[j] + width


        plt.bar(x, dt2.values.tolist()[0], width=width, label=labels[1], fc=fcs[1])
        plt.bar(x2, dt1.values.tolist()[0], width=width, label=labels[0], fc=fcs[0])
        plt.bar(x3, dt3.values.tolist()[0], width=width, label=labels[2], fc=fcs[2])
        my_y_ticks = np.arange(0, 0.7, 0.05)

        plt.xticks(range(5), ticks)
        plt.yticks(my_y_ticks)
        plt.xlabel('test missing rate')
        plt.ylabel('mae')
        plt.title('train with mr='+str(ticks[i]))
        plt.legend()
        plt.show()




def gen_four():
    x = list(range(4))
    total_width, n = 0.4, 4
    width = total_width / n

    data = pd.read_csv('data/cnnmae2.csv', header=-1)

    df = pd.DataFrame(data)[2:]
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['missing rate 0.5', 'missing rate 0.6', 'missing rate 0.7', 'missing rate 0.8', 'missing rate 0.9']
    ticks = [ 0.6, 0.7, 0.8, 0.9]
    # print(tmodel)
    for i in range(5):
        dt = df[i * 2][::2]
        print(dt)
        if i != 0:
            for j in range(len(x)):
                x[j] = x[j] + width

        plt.bar(x, dt, width=width, label=labels[i], fc=fcs[i])
        # break

    # my_x_ticks = np.arange(-5, 5, 0.5)
    my_y_ticks = np.arange(0, 0.003, 0.0001)
    # plt.xticks(my_x_ticks)
    # plt.tick_params(labels)
    plt.xticks(range(4), ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('missing rate when training model')
    plt.ylabel('mae')
    plt.legend()
    plt.show()

def gen_time():
    # name_list = ['Monday', 'Tuesday', 'Friday', 'Sunday']
    # num_list = [1.5, 0.6, 7.8, 6]
    # num_list1 = [1, 2, 3, 1]
    x = list(range(5))
    total_width, n = 0.5, 5
    width = total_width / n

    # plt.bar(x, num_list, width=width, label='boy', fc='y')
    # for i in range(len(x)):
    #     x[i] = x[i] + width
    # plt.bar(x, num_list1, width=width, label='girl', tick_label=name_list, fc='r')
    # plt.legend()
    # plt.show()

    mae=[]

    data = pd.read_csv('data/cnntime.csv',header=-1)

    df = pd.DataFrame(data)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs=['r','y','b','g','orange']
    # labels=['MR 0.5','MR 0.6','MR 0.7','MR 0.8','MR 0.9']
    labels = ['missing rate 0.5', 'missing rate 0.6', 'missing rate 0.7', 'missing rate 0.8', 'missing rate 0.9']
    ticks=[0.5,0.6,0.7,0.8,0.9]

    # print(tmodel)
    for i in range(5):
        dt = df[i*2][::2]
        print(dt)
        if i!=0:
            for j in range(len(x)):
                x[j] = x[j] + width

        plt.bar(x, dt, width=width, label=labels[i], fc=fcs[i])
        # break

    # my_x_ticks = np.arange(-5, 5, 0.5)
    my_y_ticks = np.arange(0, 0.3, 0.05)
    # plt.xticks(my_x_ticks)
    # plt.tick_params(labels)
    plt.xticks(range(5),ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('missing rate when training model')
    plt.ylabel('execution time/second')
    plt.legend(fontsize=7)
    plt.show()

def gen_line_mae():
    data = pd.read_csv('data/cnnmae2.csv', header=-1)

    df = pd.DataFrame(data)
    print(df)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['missing rate 0.5', 'missing rate 0.6', 'missing rate 0.7', 'missing rate 0.8', 'missing rate 0.9']
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    # xaxis2=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

    xaxis2=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
    plt.plot(xaxis2, df[0:1].values.tolist()[0], color='r', label='Train with missing rate 0.5', marker='o', ls='-.')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='blue', label='Train with missing rate 0.6', marker='s', ls=':')
    plt.plot(xaxis2, df[4:5].values.tolist()[0], c='orange', label='Train with missing rate 0.7', marker='*', ls='--')
    plt.plot(xaxis2, df[6:7].values.tolist()[0], c='black', label='Train with missing rate 0.8', marker='h', ls='-.')
    plt.plot(xaxis2, df[8::9].values.tolist()[0], c='g', label='Train with missing rate 0.9', marker='D', ls='-.')
    plt.legend()
    plt.xlabel('Sampling rate')
    plt.ylabel('MAE')
    plt.show()

def gen_mae2():
    data = pd.read_csv('data/cnnmae2.csv', header=-1)

    df = pd.DataFrame(data)
    print(df)
    print(df[0:1])
    xaxis2 = [0.5,  0.6, 0.7,  0.8,  0.9]

    plt.plot(xaxis2, df[0:1].values.tolist()[0], color='r', label='Train with missing rate 0.5', marker='o', ls='-')
    plt.plot(xaxis2, df[1:2].values.tolist()[0], c='blue', label='Train with missing rate 0.6', marker='s', ls=':')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='orange', label='Train with missing rate 0.7', marker='*', ls='--')
    plt.plot(xaxis2, df[3:4].values.tolist()[0], c='black', label='Train with missing rate 0.8', marker='h', ls='-.')
    plt.plot(xaxis2, df[4:5].values.tolist()[0], c='g', label='Train with missing rate 0.9', marker='D', ls='-.')
    plt.legend()
    plt.xlabel('missing rate')
    plt.ylabel('mae')
    plt.show()

def gen_line_time():
    data = pd.read_csv('backup/cnntime2.csv', header=-1)

    df = pd.DataFrame(data)
    # tmodel = np.zeros((5,5),dtype=np.int)
    fcs = ['r', 'y', 'b', 'g', 'orange']
    labels = ['Missing rate 0.5', 'Missing rate 0.6', 'missing rate 0.7', 'missing rate 0.8', 'missing rate 0.9']
    ticks = [0.5, 0.6, 0.7, 0.8, 0.9]
    xaxis2 = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    plt.plot(xaxis2, df[0:1].values.tolist()[0], color='r', label='Train with missing rate 0.5', marker='o', ls='-')
    plt.plot(xaxis2, df[2:3].values.tolist()[0], c='blue', label='Train with missing rate 0.6', marker='s', ls=':')
    plt.plot(xaxis2, df[4:5].values.tolist()[0], c='orange', label='Train with missing rate 0.7', marker='*', ls='--')
    plt.plot(xaxis2, df[6:7].values.tolist()[0], c='black', label='Train with missing rate 0.8', marker='h', ls='-.')
    plt.plot(xaxis2, df[8::9].values.tolist()[0], c='g', label='Train with missing rate 0.9', marker='D', ls='-.')
    plt.legend()
    plt.xlabel('Missing rate')
    plt.ylabel('Exetution time(second)')
    plt.show()

# gen_random2as_line()
# gen_as_train2test()
gen_random2as()
# gen_five()
# gen_line_mae()
# gen_line_time()
# gen_mae2()