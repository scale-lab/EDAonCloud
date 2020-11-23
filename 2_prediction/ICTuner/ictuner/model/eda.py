import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def drv_hist(df):
    print(len(df))
    df = df.loc[df['route_success'] == 1]
    print(len(df))
    _min = df['drv_total'].min()
    _max = df['drv_total'].max()
    df['drv_total'] = (df['drv_total'] - _min) / (_max - _min)
    plt.hist(df['drv_total'], bins=40)
    plt.xlabel('Total DRVs')
    plt.savefig('test/drv-' + str(_min) + '-' + str(_max) + '.png')
    plt.cla()

def runtime_hist(df):
    _min = df['time'].min()
    _max = df['time'].max()
    df['time'] = (df['time'] - _min) / (_max - _min)
    plt.hist(df['time'])
    plt.xlabel('Floorplanning to Routing Runtime (seconds)')
    plt.savefig('test/time-' + str(_min) + '-' + str(_max) + '.png')
    plt.cla()

def memory_hist(df):
    _min = df['memory'].min()
    _max = df['memory'].max()
    df['memory'] = (df['memory'] - _min) / (_max - _min)
    plt.hist(df['memory'], bins=20)
    plt.xlabel('Floorplanning to Routing Memory (MB)')
    plt.savefig('test/memory-' + str(_min) + '-' + str(_max) + '.png')
    plt.cla()

def power_hist(df):
    non_zero = (df['power_total'] > 0)
    plt.hist(np.log(df[non_zero]['power_total']), bins=20)
    plt.xlabel('Total Power')
    plt.savefig('test/power.png')
    plt.cla()

def area_hist(df):
    non_zero = df['area_total'] > 0
    plt.hist(df[non_zero]['area_total'])
    plt.xlabel('Total Area')
    plt.savefig('test/area.png')
    plt.cla()

if __name__ == "__main__":
    df = pd.read_csv('/home/aibrahim/ICCAD20/db.csv')

    drv_hist(df)
    runtime_hist(df)
    memory_hist(df)
    #power_hist(df)
    #area_hist(df)
    
    