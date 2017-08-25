import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_data(**kwargs):
    """
    read_data(**kwargs) reads the point cloud data for the Smart Shores project.
    
    Input
    column_names : (default: ['lon', 'lat', 'z', 'r', 'g', 'b', 'j', 'k', 'l']
    max_rows : (default: 15 000 000)
    nrows : (default: 500 000)
    skip : (default: 0)
    verbose : (default: False)
    fp : (default: '~/data/4-Vadeboncoeur/davis-bay.txt'
    """
    from pandas import read_csv as _csv
    total_rows = 57891462
    column_names = kwargs.get('column_names',
                              ['lon', 'lat', 'z', 'r',
                               'g', 'b', 'j', 'k', 'l'])
    max_rows = kwargs.get('max_rows', int(15 * 1e6))
    nrows = kwargs.get('nrows', int(5 * 1e5))
    skip = kwargs.get('skip', 0)
    verbose = kwargs.get('verbose', False)
    fp = kwargs.get('filepath', 
                    '~/data/4-Vadeboncoeur/davis-bay.txt')
    data = {}
    ctr = 0
    while skip + nrows < np.min([total_rows, max_rows]):
        data[ctr] = _csv(fp, sep=" ", header=None, 
                         skiprows=skip, nrows=nrows)
        data[ctr].columns = column_names
        skip += nrows
        ctr += 1
        if verbose:
            print('\rrows read: {}'.format(skip), end='')
    return data


def _getLonLatPairs(df):
    return df.loc[:, ['lon', 'lat']].values


def getLonLatPairs(dfDict, concat=False):
    if concat:
        return np.vstack([_getLonLatPairs(v) for v in dfDict.values()])
    else:
        return {k: _getLonLatPairs(v) for k, v in dfDict.items()}


def getUniqueLonLat(lonLatPairs):
    if isinstance(lonLatPairs, dict):
        unique_lon = np.concatenate([np.unique(v[:,0])
                                     for v in lonLatPairs.values()])
        unique_lat = np.concatenate([np.unique(v[:,1])
                                     for v in lonLatPairs.values()])
        unique_lon = np.unique(unique_lon)
        unique_lat = np.unique(unique_lat)
    elif isinstance(lonLatPairs, np.ndarray):
        unique_lon = np.unique(lonLatPairs[:,0])
        unique_lat = np.unique(lonLatPairs[:,1])
    else:
        raise Exception('Could not parse type for lonLatPairs')
    return (unique_lon, unique_lat)


def _near_equal(x, vec, condition='equality'):
    """
    _near_equal(x, vec, condition='equality') is a fancy generator that
    returns a list of tuples (j, vec[j]) corresponding to values of
    vec that are "near equal" to x. Note that one imposes exact
    equality by passing condition='equality'; or imposes a
    near-equality with tolerance th by passing condition=th. Custom
    functions mapping (number, vector, index) -> bool are also
    accepted.
    """
    if isinstance(condition, str) and (condition.lower()[0] == 'e'):
        # equality case
        cond = lambda y, v, k: (y == v[k])
    elif isinstance(condition, float):
        cond = lambda y, v, k: (np.abs(y - v[k]) <= condition)
    elif callable(condition):
        # should return bool
        cond = condition
    return np.fromiter((vec[j] for j in range(vec.size) 
                        if cond(x,vec,j)), dtype=np.float)


def _near_equal_pairs(lon, lat, lon_vec, lat_vec, th):
    npnorm = np.linalg.norm
    arr = [[x,y]
           for x in lon_vec
           for y in lat_vec
           if (npnorm([x-lon, y-lat])<th)]
    arr = np.array(arr)
    return arr


def near_equal(lon, lat, lon_vec, lat_vec, th):
    lon_ne = _near_equal(lon, lon_vec, th)
    lat_ne = _near_equal(lat, lat_vec, th)
    return _near_equal_pairs(lon, lat, lon_ne, lat_ne, th)


# Next up is to write the getNeighbours function that will get neighbours for each point.
# def _getNeighbours(lon, lat, lon_vec, lat_vec, th=None):
#     """
#     _getNeigbours(lon, lat, lon_vec, lat_vec) returns a list of 
#     the neighbouring latitudes and longitudes of (lon, lat), 
#     from the array (lon_vec, lat_vec), as determined by the
#     threshold radius th. 
#     """
#     nlon = lon_vec.size
#     nlat = lat_vec.size
#     if (th is None) and (nlon < 10000) and (nlat < 10000):
#         # this could be expensive to compute...
#         dlon = (lon_vec[1:] - lon_vec[:-1]).mean()
#         dlat = (lat_vec[1:] - lat_vec[:-1]).mean()
#         th = 3 * np.sqrt(dlon**2 + dlat**2)
#     elif th is None:
#         dlon = lon_vec[1] - lon_vec[0]
#         dlat = lat_vec[1] - lat_vec[0]
#         th = 3 * np.sqrt(dlon**2 + dlat**2)

#     return nbrs



def smallestDiff(vec, random=False, patience=1000):
    """
    smallestDiff(vec, random=True, patience=1000) approximates the
    computation of the smallest difference in a very long
    vector. Assumes a random permutation will be good enough.

    Note: random=False is good when you think the points are 
          ordered in some way.
    """
    smallest = np.inf
    p = patience
    if random:
        vec = np.random.permutation(vec)

    for j, v in enumerate(vec):
        for k, w in enumerate(vec[j+1:]):
            diff = np.abs(v - w)
            if diff < smallest:
                smallest = diff
                p = patience
            else:
                p -= 1
            if p <= 0:
                return smallest
    return smallest


def plotPointDf(df, **kwargs):
    """
    plotPointsDf(df, **kwargs), given a dataframe, makes a scatter 
    plot of the point cloud, coloured by the rgb values in 
    dataframe's columns. 
    
    Input
    df : dataframe containing point cloud data. Assumes column names
         [lon, lat, r, g, b]. 
    doScale : whether to scale the points before plotting 
              (default: True)
    figsize : the figure size (default: (10, 10))
    max_pts : the maximum allowable number of points to plot
              (default: 750 000)
    s : the point size in the scatter plot (default: 3)
    """
    # options
    doScale = kwargs.get('doScale', True)
    figsize = kwargs.get('figsize', (10,10))
    max_pts = kwargs.get('max_pts', 750000)
    s = kwargs.get('s', 3)

    # error checking
    if df.shape[0] > max_pts:
        raise Exception('too many points to plot. ' +
                        'This is controlled by the ' +
                        'max_pts parameter.')

    # formatting
    lonlat_ = df.loc[:, ['lon', 'lat']]
    # do scale? 
    if doScale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        lonlat_ = scaler.fit_transform(df.loc[:, ['lon', 'lat']])

    # colour data must be in zero-one range
    rgb_01 = df.loc[:, ['r', 'g', 'b']].values/255

    # make plot
    plt.figure(figsize=figsize)
    plt.scatter(lonlat_[:, 0], lonlat_[:, 1], c=rgb_01, s=s)
    plt.axis('equal')
    

def hist3d(arr, **kwargs):
    """
    hist3d(arr, **kwargs) plots a 3D historgram of point cloud data

    Input
    arr : the input array of which a histogram will be plotted. arr 
          should be an N-by-3 array representing values in 3-space.
    nbins : the default number of bins along each axis (default: 50)
    th : the threshold below which a bin will not be represented in
         the final histogram. (default: .01)
    figsize : the size of the output figure (default: (10, 8))
    elev : the elevation angle of the view (default : 45)
    azim : the azimuthal angle of the view (default: 30)
    cmap : the colour map used in the plot (default: viridis)
    s : the size of the points in the histogram (default: 3)
    """
    nbins = kwargs.get('nbins', 50)
    th = kwargs.get('threshold', .01)
    figsize = kwargs.get('figsize', (10,8))
    elev = kwargs.get('elev', 45)
    azim = kwargs.get('azim', 30)
    cmap = kwargs.get('cmap', 'viridis')
    s = kwargs.get('s', 3)

    H, edges = np.histogramdd(arr, bins=nbins)
    edges = np.vstack(edges).T
    edges = .5 * (edges[1:,:] + edges[:-1,:])
    edges.shape

    x = []
    y = []
    z = []
    c = []

    for j in range(nbins):
        for k in range(nbins):
            for l in range(nbins):
                v = H[j,k,l]
                if v < th:
                    continue
                else:
                    x.append(edges[j,0])
                    y.append(edges[k,1])
                    z.append(edges[l,2])
                    c.append(v)
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev, azim)
    im = ax.scatter(x, y, z, zdir='z', c=np.log(c), s=s, cmap=cmap);
    plt.colorbar(im);
    return



# # # # # # #                                          # # # # # # # 
# #                                                              # #
#     This is the good stuff that's used in the final notebook     #
# #                                                              # #
# # # # # # #                                          # # # # # # #

def kneigh(datav, n_neighbors=10, radius=1e-5):
    """
    kneigh : 
    returns an array of the n_neighbours-many nearest neighbours 
    for each row of datav (obtained by fitting a nearest neighbours
    model to the rows of datav).

    Input
    datav : dataframe.values (e.g. df.values[:, :2].shape = (500000, 2))
            where columns are (lon, lat)
    defaults:  n_clusters = 10 and radius = 1e-5 (~1 m)
    """
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=radius)
    print('fitting nearest neighbours...', end='')
    neigh.fit(datav)
    print('done.')
    print('getting k neighbors for each point...', end='')
    kneigh_dist, kneigh_ind = neigh.kneighbors(datav)
    print('done')
    return kneigh_dist, kneigh_ind

def pointFeatures(d, ind, j, dist=None):
    """
    pointFeatures(d, ind, dist, j, useSampleWeights=True)

    Input
    d : scaled data (e.g. many rows of (z,r,g,b) tuples). 
    ind : an array where each row j corresponds to the jth 
          (z,r,g,b) tuple of data_s, where each element k in
          the jth row corresponds to the kth row of d that 
          is a neighbor of j.
    dist : either None, or an array of size d.shape[0] * n_neighbors
           containing the distances from point j to each of its 
           neighbours k
    j : the row j of d for which to compute the features
    """
    ftr = d[ind[j,:],:] 
    if dist is not None:
        ftr = ftr / (np.c_[dist[j,:]]+.1)
    return ftr.ravel()

def generatePointFeatures(datav, n_neighbors=10, radius=1e-5):
    """
    generatePointFeatures

    Input
    datav = df.values where df.values.shape = (500000, 6)
            where columns are (lon, lat, z, r, g, b)
    """
    # compute the knns and associated distances
    kneigh_dist, kneigh_ind = kneigh(datav[:, :2], n_neighbors, radius)

    # Scale the (z,r,g,b) tuples
    print('scaling feature data...', end='')
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_s = scaler.fit_transform(datav[:, 2:])
    print('done.')

    # get the point features for each point
    print('fetching point features...', end='')
    # # # Important Note: # # #
    # we're not going to include kneigh_dist in the pointFeatures computation
    # because I'm not convinced it makese sense in this context. If we had 
    # used something like an adjacency sub-matrix, then maybe scaling by 
    # sample weights would make more sense. But this possibility remains 
    # open; cf. the if statement in pointFeatures(...).
    ptFtrs = [pointFeatures(data_s, kneigh_ind, j) 
              for j in range(kneigh_ind.shape[0])]
    ptFtrs = np.array(ptFtrs)
    print('done.')
    return ptFtrs

def miniBatchKMeans(X, n_clusters=20, batch_size=50000):
    """
    miniBatchKMeans
    returns trained minibatch k means on the data X
    """
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=n_clusters)
    batch = 0
    start_idx = batch * batch_size
    end_idx = (batch+1) * batch_size

    while end_idx < X.shape[0]:
        km.partial_fit(X[start_idx:end_idx, :])
        batch += 1
        start_idx = end_idx
        end_idx = (batch + 1) * batch_size

    km.partial_fit(X[start_idx:, :])
    return km

def getClusterMembership(km, X):
    """
    Return which cluster each point belongs to.
    km is a fit k-means object and X is the data where cluster
    membership is to be determined
    """
    km_ = km.transform(X)
    clusterMembership = np.argmin(km_, axis=-1)
    return clusterMembership

def makeScatterPlot(datav, clusterMembership, nClusters=None, cmap=None, **kwargs):
    """
    pretty plots the data, coloured by cluster membership
    """
    if nClusters is None:
        nClusters = np.unique(clusterMembership).size
    if cmap is None:
        cmap = plt.cm.gist_earth
    figsize = kwargs.get('figsize', (20,15))
    s = kwargs.get('s', 1)
    cb = kwargs.get('cb', True)
    nColours = cmap(np.arange(256))[np.linspace(0, 255, num=nClusters, dtype=np.int)]
    f, ax = plt.subplots(1,1,figsize=figsize)
    im = plt.scatter(datav[:,0], datav[:,1], c=nColours[clusterMembership], s=s)
    return
