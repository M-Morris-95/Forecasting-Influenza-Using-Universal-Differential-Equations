# Standard library imports
import os
import datetime as dt

# Data handling and numerical computations
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy import interpolate
import torch
from torch.utils.data import TensorDataset, DataLoader

def smooth(df, n=7):
    smoothed = pd.DataFrame(index = df.index[n:], 
                            columns = df.columns, 
                            data = np.mean(np.asarray([df[i:-(n-i)] for i in range(n)]), 0))
    return smoothed
    
def get_hhs_query_data(hhs, root = '../google_queries/', append = 'state_queries_new', ignore = [], return_all = False, smooth_after = False):
    state_pop = pd.read_csv(root + 'state_population_data_2019.csv', index_col = 0)
    state_dict =  {1:['CT', 'ME', 'MT', 'NH', 'RI', 'VT'],
                   2:['NY', 'NJ'],
                   3:['DE', 'MD', 'PA', 'VA', 'WV', 'DC'],
                   4:['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
                   5:['IL', 'IN', 'OH', 'MI', 'MN', 'WI'],
                   6:['AR', 'LA', 'NM', 'OK', 'TX'],
                   7:['IA', 'KS', 'MO', 'NE'],
                   8:['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
                   9:['AZ', 'CA', 'HI', 'NV'],
                  10:['AK', 'ID', 'OR', 'WA']}
    
    total_population = sum([state_pop[state_pop['CODE'] == code]['POP'].values[0] for code in state_dict[hhs]])
    
    dfs = []
    for code in state_dict[hhs]:
        if code not in ignore:
            population = state_pop[state_pop['CODE'] == code]['POP'].values[0]/total_population
            new_nf = population*pd.read_csv(root+append +'/'+code+'_query_data.csv', index_col=0, parse_dates=True)
            dfs.append(new_nf)
        
        
    
    cols = [d.columns for d in dfs]
    common_cols = cols[0]
    for col_list in cols[1:]:
        common_cols = common_cols.intersection(col_list)
    
    idxs = [d.index for d in dfs]
    common_idxs = idxs[0]
    for idx_list in idxs[1:]:
        common_idxs = common_idxs.intersection(idx_list)
    
    df = pd.DataFrame(index = common_idxs, columns = common_cols, data = 0)
        
    for d in dfs:
        df = df+d.loc[df.index, df.columns]

    if smooth_after:
        df = smooth(df)
        
    if return_all:
        return df, dfs
    return df    

def choose_qs(hhs_df, daily_ili, region, season, n_qs):
    queries = hhs_df[region]
    ili = daily_ili['Region '+str(region)]
    
    index = daily_ili.index.intersection(queries.index)
    queries = queries.loc[index]
    
    ili = ili.loc[index]
    
    
    dates = pd.date_range(dt.date(season-5, 10, 1), dt.date(season,10,1))

    queries_subset = queries.loc[dates].std()
    queries = queries.iloc[:, np.where(queries_subset != 0)[0]]

    corr_df = pd.DataFrame(index=queries.columns,
                 columns=['correlation'],
                 data=[pearsonr(ili.loc[dates].squeeze(), q)[0] for q in
                               queries.loc[dates].values.T])
    scores = pd.read_csv('Data/Similarity_Scores.csv', index_col=0)
    scores['correlation'] = corr_df
    # scores = scores.fillna(0)
    scores = scores.dropna()
    
    for col in scores.columns:
        scores[col] = scores[col] - scores[col].min()
        scores[col] = scores[col] / scores[col].max()
        scores[col] = 1 - scores[col]
    scores['score'] = np.sqrt(np.square(scores).sum(1))
    
    scores = scores.sort_values('score')
    
    query_choice = scores[:n_qs]
    return query_choice.index


def load_ili(location):
    location_dict = {'US':'Data/national_flu.csv',
                     'England':'Data/England_ILIrates.csv',
                     'state':'Data/state_flu.csv',
                     'hhs':'Data/hhs_flu.csv'}
    
    ili = pd.read_csv(location_dict[location], index_col = -1, parse_dates=True)
    if location == 'state' or location =='hhs':
        new_ili = pd.DataFrame()
        for region in ili['region'].unique():
            new_ili[region] = ili[ili['region'] == region]['unweighted_ili']
        ili = new_ili
        ili /= 13
        ili= ili.fillna(0)
        
    if location == 'US':
        ili[['weighted_ili']].rename(columns = {'weighted_ili':'National'})
        ili /= 13
    
    if location == 'England':
        ili['Date'] = [dt.datetime.strptime(d, '%d/%m/%Y')+dt.timedelta(days=3) for d in ili['ISOWeekStartDate'].values]
        ili = ili[['Date', 'RatePer100000']].set_index('Date')
        ili = ili.rename(columns = {'RatePer100000':'National'})


    return ili

def intepolate_ili(ili):
    dates = np.asarray([ili.index[0] + dt.timedelta(days=i) for i in
                    range((ili.index[-1] - ili.index[0]).days + 1)])

    x = np.linspace(0, 1, ili.shape[0])
    x2 = np.linspace(0, 1, dates.shape[0])
    f = interpolate.interp1d(x, ili.values, axis = 0, kind = 'cubic')
    
    daily_ili = pd.DataFrame(index=dates, columns=ili.columns, data=f(x2))
    return daily_ili

def convert_to_torch(x_train, y_train, x_test, y_test, batch_size=32, shuffle=True, dtype=torch.float32):
        x_train = torch.tensor(x_train, dtype = dtype)
        y_train = torch.tensor(y_train, dtype = dtype)
        x_test = torch.tensor(x_test, dtype = dtype)
        y_test = torch.tensor(y_test, dtype = dtype)

        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle)
        return train_loader, x_test, y_test

if __name__ == '__main__':
    tmax = 8
    suppress_outputs = True
    lag = 14
    n_regions = 10
    season = 2016
    lr = 1e-3
    n_samples = 128
    n_qs = 10
    
    root = 'checkpoints/HHS_SIR_Big_new/'      
    device = 'cpu'
    dtype=torch.float32
    

    gamma = 63
    t = torch.linspace(1,gamma, gamma, device=device)/7
    eval_pts = [0,6,13,20,27,34,40,47,54][:tmax]
    
    ili = load_ili('hhs')
    ili = intepolate_ili(ili)
    
    hhs_dict = {}
    qs_dict = {}
    
    ignore = ['AZ', 'ND', 'AL', 'RI', 'VI', 'PR']
    for i in range(1,1+n_regions):
        hhs_dict[i] = get_hhs_query_data(i, ignore=ignore, smooth_after = True)
        qs_dict[i] = choose_qs(hhs_dict, ili, i, season, n_qs)
    
        hhs_dict[i] = hhs_dict[i].loc[:, list(qs_dict[i])]
        hhs_dict[i] = hhs_dict[i].div(hhs_dict[i].max())
        