# Standard library imports
import datetime as dt

# Data handling and numerical computations
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from lib.HHS_data import *

# Setting the number of threads for PyTorch and specifying the device
torch.set_num_threads(1)

def smooth(df, n=7):
    smoothed = pd.DataFrame(index = df.index[n:], 
                            columns = df.columns, 
                            data = np.mean(np.asarray([df[i:-(n-i)] for i in range(n)]), 0))
    return smoothed

def convert_data(inputs, outputs, dates, year=2015, batch_size=32, validation=False):
    train_dates = {
        2015: {'start': dt.date(2004, 3, 24), 'end': dt.date(2015, 8, 12)},
        2016: {'start': dt.date(2004, 3, 24), 'end': dt.date(2016, 8, 11)},
        2017: {'start': dt.date(2004, 3, 24), 'end': dt.date(2017, 8, 10)},
        2018: {'start': dt.date(2004, 3, 24), 'end': dt.date(2018, 8, 9)}
    }

    test_dates = {
        2015: {'start': dt.date(2015, 10, 19), 'end': dt.date(2016, 5, 14)},
        2016: {'start': dt.date(2016, 10, 17), 'end': dt.date(2017, 5, 13)},
        2017: {'start': dt.date(2017, 10, 16), 'end': dt.date(2018, 5, 12)},
        2018: {'start': dt.date(2018, 10, 15), 'end': dt.date(2019, 5, 11)}
    }

    try:
        tr_start_idx = np.where([train_dates[year]['start'] == d for d in dates])[0][0]
    except:
        tr_start_idx = 0
    tr_end_idx = np.where([train_dates[year]['end'] == d for d in dates])[0][0]
    te_start_idx = np.where([test_dates[year]['start'] == d for d in dates])[0][0]
    te_end_idx = np.where([test_dates[year]['end'] == d for d in dates])[0][0]

    x_tr, y_tr = inputs[tr_start_idx:tr_end_idx], outputs[tr_start_idx:tr_end_idx]
    if not validation:
        x_test, y_test = inputs[te_start_idx:te_end_idx], outputs[te_start_idx:te_end_idx]
    else:
        x_test, y_test = inputs[te_start_idx:], outputs[te_start_idx:]

    x_train = [x_tr[b * batch_size:(b + 1) * batch_size] for b in range(int(np.ceil(x_tr.shape[0] / batch_size)))]
    y_train = [y_tr[b * batch_size:(b + 1) * batch_size] for b in range(int(np.ceil(y_tr.shape[0] / batch_size)))]

    return x_train, y_train, x_test, y_test

def choose_qs(queries, ili, season, n_qs):
    # Ensure the index is consistent between queries and ili
    index = ili.index.intersection(queries.index)
    queries = queries.loc[index]
    ili = ili.loc[index]

    # Create date range for the given season
    dates = pd.date_range(max(index[0].date(), dt.date(season - 5, 10, 1)), dt.date(season, 10, 1))

    # Filter out columns with zero standard deviation
    queries_subset = queries.loc[dates].std()
    queries = queries.iloc[:, np.where(queries_subset != 0)[0]]

    # Calculate correlation between ili and queries
    corr_df = pd.DataFrame(index=queries.columns,
                            columns=['correlation'],
                            data=[pearsonr(ili.loc[dates].squeeze(), q)[0] for q in queries.loc[dates].values.T])

    # Read and preprocess similarity scores
    scores = pd.read_csv('Data/Similarity_Scores.csv', index_col=0)
    scores['correlation'] = corr_df
    scores = scores.dropna()

    # Normalize scores
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores['score'] = np.sqrt(np.square(scores).sum(axis=1))

    # Sort scores and select top n_qs
    scores = scores.sort_values('score').head(n_qs)

    return scores.index

def build_data(n_qs, season, window = 35, lag = 14, batch_size=32, gamma = 28, region = 'state', ignore = ['FL'], validation = False, root = '../google_queries/', append = 'state_queries_new'):
    state_codes = {'AK':'Alaska','AL':'Alabama','AR':'Arkansas','AZ':'Arizona','CA':'California','CO':'Colorado','CT':'Connecticut','DE':'Delaware','DC':'District of Columbia','FL':'Florida','GA':'Georgia','HI':'Hawaii','ID':'Idaho','IL':'Illinois','IN':'Indiana','IA':'Iowa','KS':'Kansas','KY':'Kentucky','LA':'Louisiana','ME':'Maine','MD':'Maryland','MI':'Michigan','MN':'Minnesota','MS':'Mississippi','MO':'Missouri','MT':'Montana','NE':'Nebraska','NV':'Nevada','NH':'New Hampshire','NJ':'New Jersey','NM':'New Mexico','NY':'New York','NC':'North Carolina','ND':'North Dakota','OH':'Ohio','OK':'Oklahoma','OR':'Oregon','PA':'Pennsylvania','RI':'Rhode Island','SC':'South Carolina','SD':'South Dakota','TN':'Tennessee','TX':'Texas','UT':'Utah','VT':'Vermont','VA':'Virginia','WA':'Washington','WV':'West Virginia','WI':'Wisconsin','WY':'Wyoming'}
    
    ili = load_ili(region)
    ili = intepolate_ili(ili)

    selected_qs_dict = {}
    query_dict = {}
    for code in state_codes.keys():
        if code not in ignore:
            file_path = f'{root}{append}/{code}_query_data.csv'
            ili_state = ili[state_codes[code]]

            qs_state = pd.read_csv(file_path, index_col=0, parse_dates=True)

            qs_state = smooth(qs_state)

            selected_qs = choose_qs(qs_state, ili_state, season-1, n_qs=n_qs)
            qs_state = qs_state.loc[ili.index, selected_qs]

            qs_state = qs_state.div(qs_state.max())
            
            selected_qs_dict[code] = selected_qs
            query_dict[code] = qs_state

    inputs = []
    outputs = []
    dates = []
    for batch in range(ili.shape[0] - (window+gamma)):
        batch_inputs = []
        for code in state_codes.keys():
            if code not in ignore:
                batch_inputs.append(query_dict[code].iloc[batch:batch+window+1])
        t_ili = ili.iloc[batch:batch+window+1].copy()
        t_ili = t_ili[list(state_codes.values())]
        t_ili = t_ili.drop(columns=[state_codes[key] for key in ignore])
        t_ili.iloc[-lag:, :] = -1
        dates.append(pd.to_datetime(t_ili.index[-15]).date())
        batch_inputs.append(t_ili)
        batch_inputs = np.concatenate(batch_inputs, -1)

        o_ili = ili.iloc[batch:batch+window-lag+gamma+1].copy()
        o_ili = o_ili[list(state_codes.values())]
        o_ili = o_ili.drop(columns=[state_codes[key] for key in ignore])
        batch_outputs = o_ili.values

        inputs.append(batch_inputs)
        outputs.append(batch_outputs)
    inputs = torch.tensor(np.asarray(inputs), dtype=torch.float32)
    outputs = torch.tensor(np.asarray(outputs), dtype=torch.float32)

    x_train, y_train, x_test, y_test = convert_data(inputs, outputs, dates, year=season, batch_size=batch_size, validation=validation)

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    lag = 14
    n_regions = 49
    season = 2016
    n_qs = 10
    window = 35
    lag =14
    gamma = 28
    batch_size=32
    
    x_train, y_train, x_test, y_test = build_data(n_qs, season, window = window, gamma = gamma, lag = lag, batch_size=batch_size, region = 'state', ignore = ['FL'], root = '../google_queries/', append = 'state_queries_new')


