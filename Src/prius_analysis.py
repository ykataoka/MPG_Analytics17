import matplotlib as mpl
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json

mpl.rcParams['legend.fontsize'] = 10

#  0 timestamp			seconds		float
#  1 throttle			percent		float
#  2 brake			on/off		bool
#  3 gear			position	int
#  4 engine speed		rad/s		float
#  5 vehicle speed		m/s		float
#  6 fuel			liters		float
#  7 odometer			meters		float
#  8 latitude			degrees		float
#  9 longitude			degrees		float
# 10 not used
# 11 ev mode			on/off		bool
# 12 state of charge		percent		float
# 13 altitude (zCar)		meters		float

# initial & fixed variables
org_files = glob.glob('../Data/Laps/*')
cols = ['time[sec]',
        'throttle[percent]',
        'brake[on/off]',
        'gear[position]',
        'enginespeed[rad/s]',
        'vehiclespeed[m/s]',
        'fuel[liters]',
        'odometer[meters]',
        'latitude[degrees]',
        'longitude[degrees]',
        'notused',
        'evmode[on/off]',
        'state_of_charge[percent]',
        'altitude_zCar[meters]']


def std_normalize(A):
    """
    @desc : apply standardization
    """
    return (A - np.average(A)) / np.std(A)


def show_stats(mpgs):
    """
    @desc : print out average and std of mpg data
    """
    print 'average MPG = ' + str(np.average(mpgs))
    print 'std MPG = ' + str(np.std(mpgs))


def cluster_points(filename, N):
    """
    @desc : print out average and std of mpg data
    @param filename : filename of sample&typical data
    @param N : the number of the micro cluster
    @return centroids (num M, the attention payable points)
    """

    # read data file
    data = pd.read_json(filename)
    data.columns = cols

    # axis data
    x = data['latitude[degrees]']
    y = data['longitude[degrees]']
    z = data['altitude_zCar[meters]']

    # find the centroid (N)
    num = len(x)
    tmp = num / float(N)
    centroids = []
    for i in range(N):
        c = (x[int(i*tmp)], y[int(i*tmp)], z[int(i*tmp)])
        centroids.append(c)

    return centroids


def plot_all(files):
    """
    @desc : plot scaled & multiple dimensional data
    """
    for filename in files:

        # read data file
        data = pd.read_json(filename)
        data.columns = cols

        # read meta file
        metaname = filename.split('/')
        metaname[2] = 'LapMeta'
        metaname = '/'.join(metaname)
        f = open(metaname, 'r')
        meta = json.load(f)

        # axis data
        t = data['time[sec]']
        x = data['latitude[degrees]']
        y = data['longitude[degrees]']

        # original data
        th = data['throttle[percent]']
        e = data['evmode[on/off]']
        b = data['brake[on/off]']
        f = data['fuel[liters]']
        c = data['state_of_charge[percent]']
        v = data['vehiclespeed[m/s]'] * 3600. / (1000 * 1.6)
        z = data['altitude_zCar[meters]']

        # scaling
        s_th = std_normalize(th)
        s_e = std_normalize(e)
        s_b = std_normalize(b)
        s_f = std_normalize(f)
        s_c = std_normalize(c)
        s_v = std_normalize(v)
        s_z = std_normalize(z)

        # plot scaled value
        title = 'ALL : mpg = ' + str(meta['mpg']) + ', param = ' + str(meta['name'])
        plt.title(title)
        plt.xlabel('Time[sec]')
        plt.ylabel('Scaled')
        # plt.xlim(0,0.01)
        # plt.ylim(0, 100)
        plt.grid(True)
        plt.plot(t, s_v, label='s_velocity')
        plt.plot(t, s_b, label='s_brake')
        plt.plot(t, s_th, label='s_throttle')
        plt.plot(t, s_e, label='s_ev')
        plt.plot(t, s_f, label='s_fuel')
        plt.plot(t, s_c, label='s_charge')
        plt.plot(t, s_z, label='s_altitude')
        plt.legend()

        # save figure
        if meta['name'].find('Out') == 0:
            savename = 'result/OutLap/all_' + filename.split('/')[-1] + '.png'
        elif meta['name'].find('Fly') == 0:
            savename = 'result/FlyLap/all_' + filename.split('/')[-1] + '.png'
        elif meta['name'].find('In') == 0:
            savename = 'result/InLap/all_' + filename.split('/')[-1] + '.png'
        plt.savefig(savename, format='png', dpi=300)
        plt.clf()

        # plot 2D velocity [mile/hour]
        #    title = 'VELOCITY : mpg = ' + str(meta['mpg']) + ', param = ' + str(meta['name'])
        #    plt.title(title)
        #    plt.xlabel('Time[sec]')
        #    plt.ylabel('Velocity[mile/hour]')
        #    # plt.xlim(0,0.01)
        #    plt.ylim(0, 100)
        #    plt.grid(True)
        #    plt.plot(t, v)
        #    plt.plot(t, e*80.)
        #    plt.plot(t, c*100.)
        #    plt.show()


def cluster_analysis(files, centroids):
    """
    @desc : apply clustering, find segment, find strategy
    @param files : list of json file
    @param centroids : fixed centroids
    """
#    for filename in files:
#
#        # read data file
#        data = pd.read_json(filename)
#        data.columns = cols
#
#        # axis data
#        t = data['time[sec]']
#        x = data['latitude[degrees]']
#        y = data['longitude[degrees]']
#
#        # original data
#        th = data['throttle[percent]']
#        e = data['evmode[on/off]']
#        b = data['brake[on/off]']
#        f = data['fuel[liters]']
#        c = data['state_of_charge[percent]']
#        v = data['vehiclespeed[m/s]'] * 3600. / (1000 * 1.6)
#        z = data['altitude_zCar[meters]']
#
#        # scaling
#        s_th = std_normalize(th)
#        s_e = std_normalize(e)
#        s_b = std_normalize(b)
#        s_f = std_normalize(f)
#        s_c = std_normalize(c)
#        s_v = std_normalize(v)
#        s_z = std_normalize(z)

    # plot centroid 3D
    c_x = []
    c_y = []
    c_z = []
    for center in centroids:
        c_x.append(-center[0])
        c_y.append(center[1])
        c_z.append(center[2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(c_x, c_y, c_z, label='centroids')
    for i, (x, y, z) in enumerate(zip(c_x, c_y, c_z)):
        label = str(i)
        ax.text(x, y, z, label)

    ax.legend()
    plt.show()


# variables
OutLapFiles = []
FlyLapFiles = []
InLapFiles = []
OutMPG = []
FlyMPG = []
InMPG = []


# centroids
out_centroids = cluster_points("../Data/Laps/1481670078.886161-128-2551.json", 150)
fly_centroids = cluster_points("../Data/Laps/1481673820.703492-5271-7990.json", 150)
in_centroids = cluster_points("../Data/Laps/1481670371.83322-2756-5552.json", 150)


# split files depending on out
for filename in org_files:

    # read meta file
    metaname = filename.split('/')
    metaname[2] = 'LapMeta'
    metaname = '/'.join(metaname)
    f = open(metaname, 'r')
    meta = json.load(f)

    # update filename
    if meta['name'].find('Out') == 0:
        OutLapFiles.append(filename)
    elif meta['name'].find('Fly') == 0:
        FlyLapFiles.append(filename)
    elif meta['name'].find('In') == 0:
        InLapFiles.append(filename)

    # update mpg
    if meta['name'].find('Out') == 0:
        OutMPG.append(meta['mpg'])
    elif meta['name'].find('Fly') == 0:
        FlyMPG.append(meta['mpg'])
    elif meta['name'].find('In') == 0:
        InMPG.append(meta['mpg'])

# deal with OutLap
show_stats(OutMPG)
plot_all(OutLapFiles)
cluster_analysis(OutLapFiles, out_centroids)

# deal with FlyLap
show_stats(FlyMPG)
plot_all(FlyLapFiles)
cluster_analysis(FlyLapFiles, fly_centroids)

# deal with InLap
show_stats(InMPG)
plot_all(InLapFiles)
cluster_analysis(InLapFiles, in_centroids)
