import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class DataLoader:
    """
    A class to load and process SmartShores data.
    """
    def __init__(self, **kwargs):
        """
        self.data : the dictionary in which the data is stored.
        self.max_rows : store no more than max_rows 
                        in the data dictionary at any given time.
        self.nrows : the number of rows to read in to each key of
                     the data dictionary.
        """
        self.TOTAL_ROWS = 57891462
        self.columns = kwargs.get('column_names',
                                  ['lon', 'lat', 'z', 'r',
                                   'g', 'b', 'j', 'k', 'l'])
        self.max_rows = kwargs.get('max_rows', int(5 * 1e6))
        self.nrows = kwargs.get('nrows', int(5 * 1e5))
        self.skip = kwargs.get('skip', 0)
        self.verbose = kwargs.get('verbose', False)
        self.fp = kwargs.get('filepath', 
                             '~/data/4-Vadeboncoeur/davis-bay.txt')
        self.data = None
        self.group_number = 0
        
        
    def readData(self):
        """
        read_data() reads the point cloud data for the
        Smart Shores project.
        """
        from pandas import read_csv as _csv
        self.skip += self.group_number * self.max_rows
        data = {}
        ctr = 0
        while ctr*self.nrows < np.min([self.TOTAL_ROWS, self.max_rows]):
            data[ctr] = _csv(self.fp, sep=" ", header=None, 
                             skiprows=self.skip + ctr*self.nrows,
                             nrows=self.nrows)
            data[ctr].columns = self.columns
            ctr += 1
            if self.verbose:
                print('\rrows read: {}'.format(self.skip), end='')
        self.data = data
        self.group_number += 1
        return

    
    def _getLonLatPairs(_df):
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
        pairs = {k: _getLonLatPairs(v)
                 for k, v in self.data.items()}
        if concat:
            pairs = np.vstack(pairs.values())
        return pairs


    def uniqueLonLatPairs(self, concat=True):
        pairs = {k: np.unique(_getLonLatPairs(v), axis=0)
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
        
