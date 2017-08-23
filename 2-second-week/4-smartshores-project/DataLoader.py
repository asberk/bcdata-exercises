import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataLoader:
    """
    A class to load and process SmartShores data.
    """
    def __init__(self, **kwargs):
        """
        Optional keyword arguments
        max_rows : store no more than max_rows in the data dictionary
                   at any given time.
        self.nrows : the number of rows to read in to each key of
                     the data dictionary.

        Attributes
        TOTAL_ROWS : 57891462 is the total number of rows in the data 
                     file we were given. If this changes then this value
                     must be changed manually. 
        data : the dictionary in which the data is stored.
        """
        self.columns = kwargs.get('column_names',
                                  ['lon', 'lat', 'z', 'r',
                                   'g', 'b', 'j', 'k', 'l'])
        self.max_rows = kwargs.get('max_rows', int(5 * 1e6))
        self.nrows = kwargs.get('nrows', int(5 * 1e5))
        self.skip = kwargs.get('skip', 0)
        self.verbose = kwargs.get('verbose', False)
        self.fileName = None
        self.read_shuffled = kwargs.get('read_shuffled', True)
        if self.read_shuffled:
            self.fileName = 'davis-bay-10Mshuf.txt'
        else:
            self.fileName = 'davis-bay.txt'
        self.readDir = kwargs.get('readDir',
                                  '/home/asberk/data/4-Vadeboncoeur/')
        self.fp = self.readDir + self.fileName
        self.include_angles = kwargs.get('include_angles', False)
        if not self.include_angles:
            self.columns = self.columns[:-3]
        print('Counting total rows...', end='')
        self._total_rows()
        print(self.TOTAL_ROWS)
        self.data = None
        self.group_number = 0
        return


    def _total_rows(self):
        with open(self.fp) as fprot:
            self.TOTAL_ROWS = len(list(fprot))
        return

    
    def readData(self):
        """
        readData(self) reads the point cloud data for the
        Smart Shores project.
        Each time it is called, it loads the next batch of point cloud data.
        """
        print('Loading group {}...'.format(self.group_number), end='')
        from pandas import read_csv as _csv
        self.skip += self.group_number * self.max_rows
        data = {}
        ctr = 0
        if self.include_angles:
            # whether to exclude the angle columns
            usecols = [0,1,2,3,4,5,6,7,8]
        else:
            usecols = [0,1,2,3,4,5]
        while ctr*self.nrows < np.min([self.TOTAL_ROWS, self.max_rows]):
            data[ctr] = _csv(self.fp, sep=" ", header=None, 
                             skiprows=self.skip + ctr*self.nrows,
                             nrows=self.nrows, usecols=usecols)
            data[ctr].columns = self.columns
            ctr += 1
            if self.verbose:
                print('\rrows read: {}'.format(self.skip), end='')
        self.data = data
        self.group_number += 1
        self.lonlat = None
        self.lonlat_unique = None
        print('done!')
        return

    
    def _getLonLatPairs(self, _df):
        """
        _getLonLatPairs(_df) takes a dataframe of values
        and returns the lon and lat columns.
        """
        return _df.loc[:, ['lon', 'lat']].values


    def getLonLatPairs(self, concat=True):
        """
        getLonLatPairs(self, concat=True) returns an array or 
        dict of arrays with the lon-lat pairs from data. 
        """
        pairs = {k: self._getLonLatPairs(v)
                 for k, v in self.data.items()}
        if concat:
            pairs = np.vstack(pairs.values())
        return pairs


    def uniqueLonLatPairs(self, concat=True):
        """
        uniqueLonLatPairs(self, concat=True)
        return an array or dict of arrays whose rows contain 
        unique pairs (lon, lat). 
        """
        pairs = {k: np.unique(self._getLonLatPairs(v), axis=0)
                 for k, v in self.data.items()}
        if concat:
            pairs = np.vstack(pairs.values())
        return pairs


    def uniqueLonLat(self):
        """
        uniqueLonLat(self)
        returns unique longitude, and unique latitude - NOT 
        unique (lon,lat) pairs. For unique pairs (lon, lat), 
        see uniqueLonLatPairs.
        """
        pairsDict = self.getLonLatPairs(concat=False)
        lon_unique = [np.unique(v[:,0]) for v in pairsDict.values()]
        lat_unique = [np.unique(v[:,1]) for v in pairsDict.values()]
        lon_unique = np.unique(np.concatenate(lon_unique))
        lat_unique = np.unique(np.concatenate(lat_unique))
        return (lon_unique, lat_unique)


    def _getRgbArray(self, batch):
        return self.data[batch].loc[:, ['r', 'g', 'b']].values


    def getRgbArray(self, concat=True):
        """
        Return the array of rgb values
        """
        rgb = {k: self._getRgbArray(k) for k in self.data.keys()}
        if concat:
            rgb = np.vstack(rgb.values())
        return rgb


    def getScaledRgbArray(self):
        """
        For the current group, return the 
        np.ndarray of normalized rgb values.
        (normalization done using StandardScaler)
        """
        from sklearn.preprocessing import StandardScaler
        rgb = self.getRgbArray() / 255
        scaler = StandardScaler()
        return scaler.fit_transform(rgb)


    def nn(self, n_neighbours=10, nbrRadius=None, n_jobs=-1):
        from sklearn.neighbors import NearestNeighbors
        if self.nbrRadius is None:
            self.nbrRadius = 1e-5
        if nbrRadius is None:
            nbrRadius = self.nbrRadius
        else:
            self.nbrRadius = nbrRadius
        self.neigh = NearestNeighbors(n_neighbors=n_neighbors,
                                      radius=nbrRadius,
                                      n_jobs=n_jobs)
        if self.lonlat is None:
            self.lonlat = self.getLonLatPairs()
        self.neigh.fit(self.lonlat)
        return


    def saveKNeighboursBatch(self, batch_size, savedir='./'):
        batch_number = 0
        start_idx = batch_number * batch_size
        end_idx = (batch_number + 1) * batch_size

        if self.lonlat_unique is None:
            print('getting unique lon-lat pairs...', end='')
            self.lonlat_unique = self.uniqueLonLatPairs()
            print('done.')

        if self.neigh is None:
            print('building neighbours tree...', end='')
            self.nn()
            print('done.')
        
        while end_idx < self.lonlat_unique.shape[0]-1:
            fileName = savedir + 'radNeigh_lonlatUnique_{}_{}.npy'
            fileName = fileName.format(self.group_number, batch_number)
            queryArr = self.lonlat_unique[start_idx:end_idx]
            self._saveKNeighbours(fileName, queryArr)
            print('Batch {} complete.'.format(batch_number))
            start_idx = end_idx
            batch_number += 1
            end_idx = (batch_number + 1) * BATCH_SIZE

        queryArr = self.lonlat_unique[start_idx:]
        fileName = 'radNeigh_lonlatUnique_{}_{}.npy'
        fileName = fileName.format(self.group_number, batch_number)
        self._saveKNeighbors(fileName, queryArr)
        print('Batch {} complete.'.format(batch_number))
        return
    

    def _saveKNeighbours(self, fileName, queryArr):
        kneighBatch = self.neigh.kneighbors(queryArr,
                                            return_distance=False)
        np.save(fileName, kneighBatch)

    
