import os 
import pandas as pd
from datetime import datetime

class storage_man_swap_dataset:
    def __init__(self, filename = 'man1_storage_swap_dataset.csv'):
        self.filename = filename
        if os.path.exists(filename):
            self.df = pd.read_csv(filename)
        else: 
            # create a new dataframe
            self.create_new_df()
    
    def create_new_df(self):
        column_names = ['stor_name', 'freq (MHz)', 'precision (MHz)', 'pi (mus)', 'h_pi (mus)', 'gain (DAC units)', 'last_update']
        self.df = pd.DataFrame(columns=column_names)
        rows = []
        for idx in range(1, 13, 1): 
            row = {'stor_name': 'M1-S' + str(idx),
                   'freq (MHz)': -1, 'precision (MHz)': -1, 'pi (mus)': -1, 'h_pi (mus)': -1, 'gain (DAC units)': -1, 'last_update': datetime.now()}
            rows.append(row)
        
        # also add for the manipulate 
        row = {'stor_name': 'M1',
               'freq (MHz)': -1, 'precision (MHz)': -1, 'pi (mus)': -1, 'h_pi (mus)': -1, 'gain (DAC units)': -1, 'last_update': datetime.now()}
        rows.append(row)
        
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)
        self.df.to_csv(self.filename, index=False)
    
    # fetch the data from the csv file
    def get_freq(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name]['freq (MHz)'].values[0]
    def get_precision(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name]['precision (MHz)'].values[0]
    def get_pi(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name]['pi (mus)'].values[0]
    def get_h_pi(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name]['h_pi (mus)'].values[0]
    def get_gain(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name]['gain (DAC units)'].values[0]
    def get_last_update(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name]['last_update'].values[0]
    def get_all(self, stor_name):
        return self.df[self.df['stor_name'] == stor_name].values[0]
        
    # update the data in the csv file
    def update_freq(self, stor_name, freq):
        self.df.loc[self.df['stor_name'] == stor_name, 'freq (MHz)'] = freq
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)
    def update_precision(self, stor_name, precision):
        self.df.loc[self.df['stor_name'] == stor_name, 'precision (MHz)'] = precision
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)
    def update_pi(self, stor_name, pi):
        self.df.loc[self.df['stor_name'] == stor_name, 'pi (mus)'] = pi
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)
    def update_h_pi(self, stor_name, h_pi):
        self.df.loc[self.df['stor_name'] == stor_name, 'h_pi (mus)'] = h_pi
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)
    def update_gain(self, stor_name, gain):
        self.df.loc[self.df['stor_name'] == stor_name, 'gain (DAC units)'] = gain
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)
    def update_last_update(self, stor_name):
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)

    def update_all(self, stor_name, freq, precision, pi, h_pi, gain):
        self.df.loc[self.df['stor_name'] == stor_name, 'freq (MHz)'] = freq
        self.df.loc[self.df['stor_name'] == stor_name, 'precision (MHz)'] = precision
        self.df.loc[self.df['stor_name'] == stor_name, 'pi (mus)'] = pi
        self.df.loc[self.df['stor_name'] == stor_name, 'h_pi (mus)'] = h_pi
        self.df.loc[self.df['stor_name'] == stor_name, 'gain (DAC units)'] = gain
        self.df.loc[self.df['stor_name'] == stor_name, 'last_update'] = datetime.now()
        self.df.to_csv(self.filename, index=False)

    # add a new row to the csv file
    def append_dataset(self, stor_name, freq, precision, pi, h_pi, gain):
        new_row = {'stor_name': stor_name, 'freq (MHz)': freq, 'precision (MHz)': precision, 'pi (mus)': pi, 'h_pi (mus)': h_pi, 'gain (DAC units)': gain, 'last_update': datetime.now()}
        self.df = self.df.append(new_row, ignore_index=True)
        self.df.to_csv(self.filename, index=False)
    
    # check whether the data is up-to-date
    def is_up_to_date(self, stor, max_time_diff = 7200):
        last_update = self.get_last_update(stor)
        # Define the format of the date string
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        # Convert the string to a datetime object
        last_update_object= datetime.strptime(last_update, date_format)
        time_diff = (datetime.now() - last_update_object).total_seconds()
        return time_diff < max_time_diff
    def create_copy(self, new_filename=None):
        expts_path = ''
        # print(f"expts_path: {expts_path}")
        
        if new_filename is None:
            name, ext = os.path.splitext(os.path.basename(self.filename))
            new_filename = os.path.join(expts_path, f"{name}_test{ext}")
        else:
            new_filename = os.path.join(expts_path, new_filename)
        self.df.to_csv(new_filename, index=False)
        return new_filename
    
