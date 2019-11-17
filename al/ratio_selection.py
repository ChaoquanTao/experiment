import numpy as np
import csv
def ratio_distribution():
    out = open('data/ratio.csv', 'a', newline='')
    # writer = csv.writer(out, dialect='excel')
    srate = [0.1,0.2]
    srate2 = [0.3,0.4,0.5]
    srate3 = [.1,.2,.3,.4,.5]

    for sr in srate3:
        for r1 in np.arange(0,1.0,0.01)[1:]:
            for r2 in np.arange(0,1.0,0.01)[1:]:
                res = r1+r2-2*r1*r2
                if res>0:
                    diff = np.abs(sr - res)
                    # if(diff<0.015):
                    if (diff ==0):
                        # print(sr,r1,r2)
                        list = [1-sr,r1,r2]
                        print(list)
                        # writer = csv.writer(out, dialect='excel')
                        # writer.writerow(list)
    out.close()
ratio_distribution()