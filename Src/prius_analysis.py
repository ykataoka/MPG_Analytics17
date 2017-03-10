import matplotlib as mpl
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
from sklearn.cluster import KMeans
import matplotlib.transforms as mtrans
from matplotlib.transforms import offset_copy

from sklearn.ensemble import RandomForestRegressor

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


def rgb(minimum, maximum, value):
    """
    return RGB value based on the min and max value
    """
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


def std_normalize(A):
    """
    @desc : apply standardization
    """
    return (A - np.average(A)) / np.std(A)


def show_stats(mpgs, files):
    """
    @desc : print out average and std of mpg data
    """
    times = []
    for filename in files:
        data = pd.read_json(filename)
        data.columns = cols

        # x,y data
        t = data['time[sec]']
        times.append(float(t.iloc[-1]))
    print 'average time = ' + str(np.average(times))
    print 'std time = ' + str(np.std(times))
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
        c = [x[int(i*tmp)], y[int(i*tmp)], z[int(i*tmp)]]
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
        g = data['gear[position]']
        e = data['evmode[on/off]']
        b = data['brake[on/off]']
        f = data['fuel[liters]']
        c = data['state_of_charge[percent]']
        v = data['vehiclespeed[m/s]'] * 1.6
        z = data['altitude_zCar[meters]']

        # scaling
        s_th = std_normalize(th)
        s_g = g
        s_e = e
        s_b = b
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
        plt.plot(t, s_g, label='s_gear')
        plt.plot(t, s_e, label='s_ev')
        plt.plot(t, s_f, label='s_fuel')
        plt.plot(t, s_c, label='s_charge')
        plt.plot(t, s_z, label='s_altitude', ls='--')
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


def plot_velocity(files):
    """
    @desc : apply clustering, find segment, find strategy
    @param files : list of json file
    @param centroids : fixed centroids
    @param lapmode : in, out, fly
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

        # original data
        f = data['fuel[liters]']
        v = data['vehiclespeed[m/s]'] * 1.6  # v[kmh] / 1.6 = v[mph]
#        v = data['vehiclespeed[m/s]'] * 3600. /(1000. * 1.6)  # v[kmh] / 1.6 = v[mph]
        z = data['altitude_zCar[meters]']
#        print data['vehiclespeed[m/s]']
#        print v
#        raw_input()

        # plot 2D velocity [mile/hour]
        title = 'VELOCITY : mpg = ' + str(meta['mpg']) + ', param = ' + str(meta['name'])
        plt.title(title)
        plt.xlabel('Time[sec]')
        plt.ylabel('Altitude[m] / Velocity[mile/h]')
        # plt.xlim(0,0.01)
        plt.ylim(-5, 100)
        plt.grid(True)
#        plt.plot(t, v, label='converted v_conv[mile/h] = v_org[m/s] * 3600. / (1000. * 1.6)')
        plt.plot(t, v, label='velocity[mile/h]')
        plt.plot(t, z, label='altitude')
        plt.legend()

        if meta['name'].find('Out') == 0:
            savename = 'result/OutLap/velocity_' + filename.split('/')[-1] + '.png'
        elif meta['name'].find('Fly') == 0:
            savename = 'result/FlyLap/velocity_' + filename.split('/')[-1] + '.png'
        elif meta['name'].find('In') == 0:
            savename = 'result/InLap/velocity_' + filename.split('/')[-1] + '.png'
        plt.savefig(savename, format='png', dpi=300)
        plt.clf()
        
#        plt.show()


def show_3D_plot(centroids):
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


def ev_mode_analytics(datas, centroids, num_c, lapmode, datamode):
    """
    @desc : analysis and visualization on ev switching
    @param datas : data enhanced by cluster label
    @param num_c : number of cluster
    @param centroids : fixed centroids
    """
    # loop for each cluster
    ev_mode_use = pd.DataFrame([], columns=['clusterID', 'ave', 'std'])
    for c in range(num_c):
        tmp_ev_ave = []
        for data in datas:  # loop for each cluster
            ev_l = data[data['label'] == c]['evmode[on/off]']
            ev_l_ave = np.average(ev_l)
            tmp_ev_ave.append(ev_l_ave)
        tmp = pd.DataFrame([[c, np.average(tmp_ev_ave), np.std(tmp_ev_ave)]],
                           columns=['clusterID', 'ave', 'std'])
        ev_mode_use = ev_mode_use.append(tmp)

    # plot cluster point
    # average : radius, vairance : color
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    radius = np.array(ev_mode_use['ave'])
    min_std, max_std = min(ev_mode_use['std']), max(ev_mode_use['std'])
    Color_c = ["#%02x%02x%02x" % rgb(min_std, max_std, value)
               for value in ev_mode_use['std']]
    X = np.array(centroids)[:, 0]
    Y = -np.array(centroids)[:, 1]
    ax.scatter(X, Y, c=Color_c, s=radius*300)

    # add centroids by text
    trans_offset = mtrans.offset_copy(ax.transData, fig=fig,
                                      x=-0.25, y=0.00, units='inches')
    for i, (x, y) in enumerate(np.array(centroids)[:, 0:2]):
        label = str(i)
        ax.text(x, -y, label, transform=trans_offset)
    ax.set_title('EV mode switching behavior')
    plt.xlim(38.159, 38.159+0.008)
    plt.ylim(122.45+0.002, 122.45+0.015)
    plt.grid(True)
    savename = datamode + '_ev_' + lapmode + '.png'
    plt.savefig(savename, format='png', dpi=300)
    plt.clf()
    #plt.show()


def brake_analytics(datas, centroids, num_c, lapmode, datamode):
    """
    @desc : analysis and visualization on brake use
    @param datas : data enhanced by cluster label
    @param num_c : number of cluster
    @param centroids : fixed centroids
    """
    # loop for each cluster
    brake_use = pd.DataFrame([], columns=['clusterID', 'ave', 'std'])
    for c in range(num_c):
        tmp_brake_ave = []
        for data in datas:  # loop for each cluster
            brake_l = data[data['label'] == c]['brake[on/off]']
            brake_l_ave = np.average(brake_l)
            tmp_brake_ave.append(brake_l_ave)
        tmp = pd.DataFrame([[c, np.average(tmp_brake_ave),
                             np.std(tmp_brake_ave)]],
                           columns=['clusterID', 'ave', 'std'])
        brake_use = brake_use.append(tmp)

    # plot cluster point
    # average : radius, vairance : color
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    radius = np.array(brake_use['ave'])
    min_std, max_std = min(brake_use['std']), max(brake_use['std'])
    Color_c = ["#%02x%02x%02x" % rgb(min_std, max_std, value)
               for value in brake_use['std']]
    X = np.array(centroids)[:, 0]
    Y = -np.array(centroids)[:, 1]
    ax.scatter(X, Y, c=Color_c, s=radius*300)

    # add centroids by text
    trans_offset = mtrans.offset_copy(ax.transData, fig=fig,
                                      x=-0.25, y=0.00, units='inches')
    for i, (x, y) in enumerate(np.array(centroids)[:, 0:2]):
        label = str(i)
        ax.text(x, -y, label, transform=trans_offset)
    ax.set_title('brake use behavior')
    plt.xlim(38.159, 38.159+0.008)
    plt.ylim(122.45+0.002, 122.45+0.015)
    plt.grid(True)
    savename = datamode + '_brake_' + lapmode + '.png'
    plt.savefig(savename, format='png', dpi=300)
    plt.clf()
#    plt.show()


def gear_analytics(datas, centroids, num_c, lapmode, datamode):
    """
    @desc : analysis and visualization on gear switching
    @param datas : data enhanced by cluster label
    @param num_c : number of cluster
    @param centroids : fixed centroids
    """
    # loop for each cluster
    gear_use = pd.DataFrame([], columns=['clusterID', 'ave', 'std'])
    for c in range(num_c):
        tmp_gear_ave = []
        for data in datas:  # loop for each cluster
            gear_l = data[data['label'] == c]['gear[position]']
            gear_l_ave = np.average(gear_l - 2)  # subtract offset position
            tmp_gear_ave.append(gear_l_ave)
        tmp = pd.DataFrame([[c, np.average(tmp_gear_ave),
                             np.std(tmp_gear_ave)]],
                           columns=['clusterID', 'ave', 'std'])
        gear_use = gear_use.append(tmp)

    # plot cluster point
    # average : radius, vairance : color
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    radius = np.array(gear_use['ave'])
    min_std, max_std = min(gear_use['std']), max(gear_use['std'])
    Color_c = ["#%02x%02x%02x" % rgb(min_std, max_std, value)
               for value in gear_use['std']]
    X = np.array(centroids)[:, 0]
    Y = -np.array(centroids)[:, 1]
    ax.scatter(X, Y, c=Color_c, s=radius*300)

    # add centroids by text
    trans_offset = mtrans.offset_copy(ax.transData, fig=fig,
                                      x=-0.25, y=0.00, units='inches')
    for i, (x, y) in enumerate(np.array(centroids)[:, 0:2]):
        label = str(i)
        ax.text(x, -y, label, transform=trans_offset)
    ax.set_title('gear use behavior')
    plt.xlim(38.159, 38.159+0.008)
    plt.ylim(122.45+0.002, 122.45+0.015)
    plt.grid(True)
    savename = datamode + '_gear_' + lapmode + '.png'
    plt.savefig(savename, format='png', dpi=300)
    plt.clf()
#    plt.show()


def throttle_analytics(datas, centroids, num_c, lapmode, datamode):
    """
    @desc : analysis and visualization on throttle use
    @param datas : data enhanced by cluster label
    @param num_c : number of cluster
    @param centroids : fixed centroids
    throttle = thtl
    """
    # loop for each cluster
    thtl_use = pd.DataFrame([], columns=['clusterID', 'ave', 'std'])
    for c in range(num_c):
        tmp_thtl_ave = []
        for data in datas:  # loop for each cluster
            thtl_l = data[data['label'] == c]['throttle[percent]']
            thtl_l_ave = np.average(thtl_l)  # subtract offset position
            tmp_thtl_ave.append(thtl_l_ave)
        tmp = pd.DataFrame([[c, np.average(tmp_thtl_ave),
                             np.std(tmp_thtl_ave)]],
                           columns=['clusterID', 'ave', 'std'])
        thtl_use = thtl_use.append(tmp)

    # plot cluster point
    # average : radius, vairance : color
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    radius = np.array(thtl_use['ave'])
    min_std, max_std = min(thtl_use['std']), max(thtl_use['std'])
    Color_c = ["#%02x%02x%02x" % rgb(min_std, max_std, value)
               for value in thtl_use['std']]
    X = np.array(centroids)[:, 0]
    Y = -np.array(centroids)[:, 1]
    ax.scatter(X, Y, c=Color_c, s=radius*300)

    # add centroids by text
    trans_offset = mtrans.offset_copy(ax.transData, fig=fig,
                                      x=-0.25, y=0.00, units='inches')
    for i, (x, y) in enumerate(np.array(centroids)[:, 0:2]):
        label = str(i)
        ax.text(x, -y, label, transform=trans_offset)
    ax.set_title('throttle use behavior')
    plt.xlim(38.159, 38.159+0.008)
    plt.ylim(122.45+0.002, 122.45+0.015)
    plt.grid(True)
    savename = datamode + '_thtl_' + lapmode + '.png'
    plt.savefig(savename, format='png', dpi=300)
    plt.clf()
#    plt.show()


def feature_generation(datas, centroids, num_c):
    """
    @desc : analysis and visualization on ev switching
    @param datas : data enhanced by cluster label
    @param centroids : fixed centroids
    @param num_c : number of cluster
    """
    # output feature
    E, B, G, T, X = [], [], [], [], []

    # E(ev)        = [data_num x 150] matrix
    # B(brake)     = [data_num x 150] matrix
    # G(gear)      = [data_num x 150] matrix
    # T(throttle)  = [data_num x 150] matrix
    # X =[E,B,G,T] = [data_num x 600] matrix

    # loop for each cluster
    # EV mode [1:EV ON, 0:EV OFF]
    for c in range(num_c):
        tmp_ev_ave = []
        for data in datas:  # loop for each cluster
            ev_l = data[data['label'] == c]['evmode[on/off]']
            ev_l_ave = np.average(ev_l)  # take average at each cluster
            tmp_ev_ave.append(ev_l_ave)
        E.append(tmp_ev_ave)
    E = np.array(E).transpose()  # 9 x 150 matrix when 'high'
    E_f_name = ['E_' + str(i) for i in range(150)]

    # Brake mode [1:Brake ON, 0:Brake OFF]
    for c in range(num_c):
        tmp_brake_ave = []
        for data in datas:  # loop for each cluster
            brake_l = data[data['label'] == c]['brake[on/off]']
            brake_l_ave = np.average(brake_l)
            tmp_brake_ave.append(brake_l_ave)
        B.append(tmp_brake_ave)
    B = np.array(B).transpose()  # 9 x 150 matrix when 'high'
    B_f_name = ['B_' + str(i) for i in range(150)]

    # Gear mode [1:Drive, 0:Neutral]
    for c in range(num_c):
        tmp_gear_ave = []
        for data in datas:  # loop for each cluster
            gear_l = data[data['label'] == c]['gear[position]']
            gear_l_ave = np.average(gear_l - 2)  # scaling
            tmp_gear_ave.append(gear_l_ave)
        G.append(tmp_gear_ave)
    G = np.array(G).transpose()  # 9 x 150 matrix when 'high'
    G_f_name = ['G_' + str(i) for i in range(150)]

    # Throttle mode [1:100%, 0:0%]
    for c in range(num_c):
        tmp_thtl_ave = []
        for data in datas:  # loop for each cluster
            thtl_l = data[data['label'] == c]['throttle[percent]']
            thtl_l_ave = np.average(thtl_l)  # subtract offset position
            tmp_thtl_ave.append(thtl_l_ave)
        T.append(tmp_thtl_ave)
    T = np.array(T).transpose()  # 9 x 150 matrix when 'high'
    T_f_name = ['T_' + str(i) for i in range(150)]

    X = np.concatenate([E, B, G, T], axis=1)
    feature_name = E_f_name + B_f_name + G_f_name + T_f_name
    print feature_name

    return [X, feature_name]


def label_creation(files):
    """
    @desc create label data, list of mpg
    """
    Y = []  # list of mpg
    for filename in files:
        # read meta file
        metaname = filename.split('/')
        metaname[2] = 'LapMeta'
        metaname = '/'.join(metaname)
        f = open(metaname, 'r')
        meta = json.load(f)
        Y.append(meta['mpg'])

    return Y


def supervised_learning(X, Y):
    """
    @desc apply RandomForest and output Feature Importance
    """
    rf = RandomForestRegressor()
    rf.fit(X, Y)
    pred_Y = rf.predict(X)
    print 'predicted mpg = ', pred_Y
    print '     true mpg = ', Y
    print 'MSE = ', np.average(abs(pred_Y - Y))
    print 'R^2 score = ', rf.score(X, Y)

    return rf.feature_importances_


def show_feature_importance(feature_name, Imp):
    """
    @desc List up feature_name based on ranking
    """
    ranking = sorted(zip(map(lambda x: round(x, 4), Imp), feature_name),
                     reverse=True)

    topK = 20
    table = []
    for i in range(topK):
        tmp = [ranking[i][0], ranking[i][1]]
        table.append(tmp)

    print pd.DataFrame(table, columns=['Importance', 'Feature_name'])
    

def cluster_analysis(files, centroids, lapmode, datamode):
    """
    @desc : apply clustering, find segment, find strategy
    @param files : list of json file
    @param centroids : fixed centroids
    @param lapmode : in, out, fly
    """

    datas = []  # list of pandas data
    num_c = len(centroids)

    """
    clustering : identify the cluster id for each file's point
    @output datas has the cluster id(label) column
    """
    for filename in files:

        # read data file
        data = pd.read_json(filename)
        data.columns = cols

        # x,y data
        t = data['time[sec]']
        x = data['latitude[degrees]']
        y = data['longitude[degrees]']
        z = data['altitude_zCar[meters]']
        X = pd.DataFrame(np.transpose([np.array(x), np.array(y), np.array(z)]),
                         columns=['x', 'y', 'z'])
        X.index = t

        # clustering with the given centroids
        Xc = np.array(X)
        Xc = Xc[:, 0:2]
        init_c = np.array(centroids)
        init_c = init_c[:, 0:2]
        kmeans = KMeans(n_clusters=num_c, init=init_c).fit(Xc)
        label_c = kmeans.labels_
        tmp = pd.DataFrame(label_c, columns=['label'])
        data = pd.concat([data, tmp], axis=1)  # add to the data
        datas.append(data)

    """
    Plot Altitude at Each Cluster ID
    """
    # if datamode == 'high':
    #     filename = files[1].split('/')[-1]
    #     tmp = datas[1]
    #     z = tmp['altitude_zCar[meters]']
    #     v = tmp['vehiclespeed[m/s]'] * 1.6  # !this is trick for true mph
    #     t = tmp['label']
    #     title = 'altitude vs label ID'
    #     plt.title(title)
    #     plt.xlabel('Label ID')
    #     plt.ylabel('Altitude [m] / vehicle speed[mile/h]')
    #     # plt.xlim(0,0.01)
    #     plt.ylim(-5, 100)
    #     plt.plot(t, z, label='altitude')
    #     plt.plot(t, v, label='converted v[mile/h]')
    #     plt.grid(True)
    #     plt.legend()
    #     savename = 'result/' + lapmode + '_altitude_labelID_' + filename + '.png'
    #     plt.savefig(savename, format='png', dpi=300)
    #     plt.clf()

    """
    Behavior Analytics at Each Cluster ID
    """
    #     ev_mode_analytics(datas, centroids, num_c, lapmode, datamode)
    #     brake_analytics(datas, centroids, num_c, lapmode, datamode)
    #     gear_analytics(datas, centroids, num_c, lapmode, datamode)
    #     throttle_analytics(datas, centroids, num_c, lapmode, datamode)

    """
    Strategy Importance Analysis
    """
    X, feature_name = feature_generation(datas, centroids, num_c)
    Y = label_creation(files)
    Imp = supervised_learning(X, Y)  # Imp = feature_importances_
    show_feature_importance(feature_name, Imp)


# variables
OutLapFiles = []
FlyLapFiles = []
InLapFiles = []
HighOutLapFiles = []
HighFlyLapFiles = []
HighInLapFiles = []
LowOutLapFiles = []
LowFlyLapFiles = []
LowInLapFiles = []
OutMPG = []
FlyMPG = []
InMPG = []


# centroids
out_centroids = cluster_points(
    "../Data/Laps/1481670078.886161-128-2551.json", 150)
fly_centroids = cluster_points(
    "../Data/Laps/1481673820.703492-5271-7990.json", 150)
in_centroids = cluster_points(
    "../Data/Laps/1481670371.83322-2756-5552.json", 150)


# split files depending on out
for filename in org_files:

    # read meta file
    metaname = filename.split('/')
    metaname[2] = 'LapMeta'
    metaname = '/'.join(metaname)
    f = open(metaname, 'r')
    meta = json.load(f)

    # update filename 'All'
    if meta['name'].find('Out') != -1:
        OutLapFiles.append(filename)
    elif meta['name'].find('Fly') != -1:
        FlyLapFiles.append(filename)
    elif meta['name'].find('In') != -1:
        InLapFiles.append(filename)

    # update filename 'High'
    if meta['name'].find('Out') != -1 and meta['name'].find('High') != -1:
        HighOutLapFiles.append(filename)
    elif meta['name'].find('Fly') != -1 and meta['name'].find('High') != -1:
        HighFlyLapFiles.append(filename)
    elif meta['name'].find('In') != -1 and meta['name'].find('High') != -1:
        HighInLapFiles.append(filename)

    # update filename 'Low'
    if meta['name'].find('Out') != -1 and meta['name'].find('Low') != -1:
        LowOutLapFiles.append(filename)
    elif meta['name'].find('Fly') != -1 and meta['name'].find('Low') != -1:
        LowFlyLapFiles.append(filename)
    elif meta['name'].find('In') != -1 and meta['name'].find('Low') != -1:
        LowInLapFiles.append(filename)

    # update mpg
    if meta['name'].find('Out') != -1:
        OutMPG.append(meta['mpg'])
    elif meta['name'].find('Fly') != -1:
        FlyMPG.append(meta['mpg'])
    elif meta['name'].find('In') != -1:
        InMPG.append(meta['mpg'])


# plot 3D image
# show_3D_plot(out_centroids)
# show_3D_plot(fly_centroids)
# show_3D_plot(in_centroids)

# plot velocity
# plot_velocity(HighOutLapFiles)
# plot_velocity(HighFlyLapFiles)
# plot_velocity(HighInLapFiles)

# deal with OutLap
print "outlap data analytics"
show_stats(OutMPG, OutLapFiles)
print 'all out : ', OutLapFiles
cluster_analysis(OutLapFiles, out_centroids, 'out', 'all')
# print 'high out : ', HighOutLapFiles
# cluster_analysis(HighOutLapFiles, out_centroids, 'out', 'high')
# print 'low out : ', LowOutLapFiles
# cluster_analysis(LowOutLapFiles, out_centroids, 'out', 'low')
# plot_all(OutLapFiles)

# deal with FlyLap
print "flylap data analytics"
show_stats(FlyMPG, FlyLapFiles)
print 'all fly : ', FlyLapFiles
cluster_analysis(FlyLapFiles, fly_centroids, 'fly', 'all')
# print 'high fly : ', HighFlyLapFiles
# cluster_analysis(HighFlyLapFiles, fly_centroids, 'fly', 'high')
# print 'low fly : ', LowFlyLapFiles
# cluster_analysis(LowFlyLapFiles, fly_centroids, 'fly', 'low')
# plot_all(FlyLapFiles)

# deal with InLap
print "inlap data analytics"
show_stats(InMPG, InLapFiles)
print 'all in : ', InLapFiles
cluster_analysis(InLapFiles, in_centroids, 'in', 'all')
# print 'high in : ', HighInLapFiles
# cluster_analysis(HighInLapFiles, in_centroids, 'in', 'high')
# print 'low in : ', LowInLapFiles
# cluster_analysis(LowInLapFiles, in_centroids, 'in', 'low')
# plot_all(InLapFiles)
