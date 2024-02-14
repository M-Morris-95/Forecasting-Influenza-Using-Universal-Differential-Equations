import pandas as pd
import torch
import numpy as np
import datetime as dt

ili=None
window=5
gamma=6
batch_size=12
dtype=torch.float32
split = 10
rescale=None
year = 2016

def load_queries(data_season, country, n_queries=100):

    country_code = 'UK' if country == 'England' else 'US'
    ILI = pd.read_csv('/home/mimorris/Dropbox/ONNX_Forecaster_2/data/ILI_rates_'+country_code + '_thursday_cubic_interpolation.csv', index_col=0, parse_dates=True)
    Qs = pd.read_csv('/home/mimorris/Dropbox/ONNX_Forecaster_2/data/'+country_code+'_Qs_small_7day_avg.csv', index_col=0, parse_dates=True)
    Qs = Qs[~Qs.index.duplicated(keep='first')].sort_index()
    Qs = Qs.rename(columns={query: query.replace('+', ' ').replace(',', ' ') for query in Qs.columns})
    Qs = Qs.rename(columns={query: ' '.join(sorted(query.split(' '))) for query in Qs.columns})
    Qs = Qs.loc[:, ~Qs.columns.duplicated()]
    Qs = Qs[np.invert(Qs.isna().all(1))]
    Qs = Qs[Qs.columns[np.where(np.max(Qs, 0) != 0)]]

    # find dates to compare Qs and ILI  
    comparison_dates = pd.date_range(dt.date(data_season-3, 1, 1), dt.date(data_season, 1, 1))
    corr = (Qs.loc[comparison_dates]).corrwith((ILI.loc[comparison_dates, 'wILI']))
    sim_score = pd.read_csv('/home/mimorris/Dropbox/ONNX_Forecaster_2/data/UK_Similarity_Scores.csv', index_col = 0)
    corr = corr/corr.max()
    sim_score = sim_score/sim_score.max()
    comparison_df = pd.DataFrame(index = corr.index, data=np.asarray([sim_score.loc[corr.index].values.squeeze(), corr.values]).T, columns = ['sim_score', 'corr'])
    comparison_df['rating'] = np.sqrt(np.square(comparison_df).sum(1))
    selected_queries = comparison_df.sort_values('rating').index[::-1]

    index_new = np.sort(list(set(ILI.index).intersection(Qs.index)))
    Qs = Qs.loc[index_new, selected_queries[:n_queries]]
    
    return Qs


def data(ili=None, window=12, gamma=6, batch_size=10, dtype=torch.float32, split = None, rescale=None, year = 2016, data_season = 2015, n_queries = 0, lag=14, country='England'):
    
    test_dates = {
        2004: [dt.date(2004, 3, 24), dt.date(2004, 8, 12), dt.date(2004, 10, 19), dt.date(2005, 5, 14)],
        2005: [dt.date(2004, 3, 24), dt.date(2005, 8, 11), dt.date(2005, 10, 19), dt.date(2006, 5, 14)],
        2006: [dt.date(2004, 3, 24), dt.date(2006, 8, 16), dt.date(2006, 10, 19), dt.date(2007, 5, 14)],
        2007: [dt.date(2004, 3, 24), dt.date(2007, 8, 15), dt.date(2007, 10, 19), dt.date(2008, 5, 14)],
        2008: [dt.date(2004, 3, 24), dt.date(2008, 8, 14), dt.date(2008, 10, 19), dt.date(2009, 5, 14)],
        2009: [dt.date(2004, 3, 24), dt.date(2009, 8, 13), dt.date(2009, 10, 19), dt.date(2010, 5, 14)],
        2010: [dt.date(2004, 3, 24), dt.date(2010, 8, 12), dt.date(2010, 10, 19), dt.date(2011, 5, 14)],
        2011: [dt.date(2004, 3, 24), dt.date(2011, 8, 11), dt.date(2011, 10, 19), dt.date(2012, 5, 14)],
        2012: [dt.date(2004, 3, 24), dt.date(2012, 8, 16), dt.date(2012, 10, 19), dt.date(2013, 5, 14)],
        2013: [dt.date(2004, 3, 24), dt.date(2013, 8, 15), dt.date(2013, 10, 19), dt.date(2014, 5, 14)],
        2014: [dt.date(2004, 3, 24), dt.date(2014, 8, 14), dt.date(2014, 10, 19), dt.date(2015, 5, 14)],
        2015: [dt.date(2004, 3, 24), dt.date(2015, 8, 13), dt.date(2015, 11, 1),  dt.date(2016, 4, 17)],
        2016: [dt.date(2004, 3, 24), dt.date(2016, 8, 11), dt.date(2016, 10, 29), dt.date(2017, 4, 16)],
        2017: [dt.date(2004, 3, 24), dt.date(2017, 8, 10), dt.date(2017, 10, 28), dt.date(2018, 4, 15)],
        2018: [dt.date(2004, 3, 24), dt.date(2018, 8, 9 ), dt.date(2018, 10, 27), dt.date(2019, 4, 14)]}


        
    if ili == None:
        try:
            ili = pd.read_csv('Data/national_flu.csv', index_col = -1, parse_dates=True)['weighted_ili'].iloc[300:]
            ili = pd.DataFrame(index = ili.index, columns = ['weighted_ili'], data = ili.values)
        except:
            ili = pd.read_csv('/Users/michael/Documents/datasets/Data/ILI_rates/national_flu.csv', index_col = -1, parse_dates=True)['weighted_ili'].iloc[300:]
            ili = pd.DataFrame(index = ili.index, columns = ['weighted_ili'], data = ili.values)
    
    if country == 'England':
        ili = pd.read_csv('/home/mimorris/Datasets/Flu/ILI_rates_UK_thursday_cubic_interpolation.csv', index_col = 0, parse_dates = True)
        ili = pd.DataFrame(index = ili.index.values.reshape(-1, 7)[:, 0], columns = ['weighted_ili'], data = ili.values.reshape(-1, 7)[:, 0])
        
    ili.index = ili.index + dt.timedelta(days = 3)
    ili_max = ili.values.max()
    if rescale:
        ili/=ili_max 


    x_tr = [ili.loc[pd.Timestamp(date-dt.timedelta(weeks = window)):pd.Timestamp(date)].values for date in pd.date_range(test_dates[year][0], test_dates[year][1], freq=dt.timedelta(weeks=1))]
    x_te = [ili.loc[pd.Timestamp(date-dt.timedelta(weeks = window)):pd.Timestamp(date)].values for date in pd.date_range(test_dates[year][2], test_dates[year][3], freq=dt.timedelta(weeks=1))]

    y_tr = [ili.loc[pd.Timestamp(date-dt.timedelta(weeks = window)):pd.Timestamp(date + dt.timedelta(weeks = gamma))].values for date in pd.date_range(test_dates[year][0], test_dates[year][1], freq=dt.timedelta(weeks=1))]        
    y_te = [ili.loc[pd.Timestamp(date-dt.timedelta(weeks = window)):pd.Timestamp(date + dt.timedelta(weeks = gamma))].values for date in pd.date_range(test_dates[year][2], test_dates[year][3], freq=dt.timedelta(weeks=1))]

    x_tr = torch.tensor(np.asarray(x_tr), dtype = dtype)
    y_tr = torch.tensor(np.asarray(y_tr), dtype = dtype)
    x_test = torch.tensor(np.asarray(x_te), dtype = dtype)
    y_test = torch.tensor(np.asarray(y_te), dtype = dtype)

    if country != 'England':
        x_tr = x_tr[:, :-1, :]
        y_tr = y_tr[:, :-1, :]

    x_train = [x_tr[i*batch_size:(i+1)*batch_size] for i in range(int(np.ceil(x_tr.shape[0] / batch_size)))]
    y_train = [y_tr[i*batch_size:(i+1)*batch_size] for i in range(int(np.ceil(x_tr.shape[0] / batch_size)))]

    if n_queries != 0:
        Qs = load_queries(data_season, country, n_queries=n_queries)
        Qs = Qs/np.tile(np.asarray(Qs.max(0)), (Qs.shape[0], 1))
        Qs = Qs.loc[ili.index[0]:ili.index[-1]]

        q_tr = [Qs.loc[pd.Timestamp(date-dt.timedelta(weeks = window)):pd.Timestamp(date+dt.timedelta(days=lag))].values for date in pd.date_range(test_dates[year][0], test_dates[year][1], freq=dt.timedelta(weeks=1))]
        q_tr = [q_tr[i*batch_size:(i+1)*batch_size] for i in range(int(np.ceil(len(q_tr) / batch_size)))]
        q_tr = [torch.tensor(np.asarray(q), dtype=torch.float32) for q in q_tr]
        
        q_te = [Qs.loc[pd.Timestamp(date-dt.timedelta(weeks = window)):pd.Timestamp(date+dt.timedelta(days=lag))].values for date in pd.date_range(test_dates[year][2], test_dates[year][3], freq=dt.timedelta(weeks=1))]
        q_te = torch.tensor(np.asarray(q_te), dtype = torch.float32)
        
        return x_train, y_train, x_test, y_test, q_tr, q_te, ili_max
        
    return x_train, y_train, x_test, y_test, ili_max