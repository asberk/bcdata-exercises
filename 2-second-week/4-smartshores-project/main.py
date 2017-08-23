import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import DataLoader
from helper_functions import hist3d


def computeFeatureVector():
    
    return
    


def main():
    filepath = '/home/asberk/data/4-Vadeboncoeur/davis-bay.txt'
    dl = DataLoader(filepath=filepath)

    dl.readData()

    lonlat = dl.getLonLatPairs()
    lon_unique, lat_unique = dl.uniqueLonLat()
    lonlat_unique = dl.uniqueLonLatPairs()
    scaledRgb = dl.getScaledRgbArray()

    computeFeatureVector()

    return

if __name__ == '__main__':
    main()
