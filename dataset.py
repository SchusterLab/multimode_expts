import os
from datetime import datetime

import pandas as pd

class mm_dataset:
    def __init__(self) -> None:
        raise NotImplementedError('This has been renamed to MMDataset')
    
class storage_man_swap_dataset:
    def __init__(self) -> None:
        raise NotImplementedError('This has been renamed to StorageManSwapDataset')
    
class  floquet_storage_swap_dataset:
    def __init__(self) -> None:
        raise NotImplementedError('This has been renamed to FloquetStorageSwapDataset')


class MMDataset:
    def __init__(self, filename):
        self.filename = filename
        if os.path.exists(filename):
            self.df = pd.read_csv(filename)
        else:
            # create a new dataframe
            self.create_new_df()
        self.indexing_row_name = self.get_columns()[0]
        # The first column is the name of each row, this variable stores the name for the name of each row
        if self.get_columns()[-1] == 'last_update':
            self.add_timestamp = True

    def create_new_df(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses by calling self.create_new_df(column_names, rows)."
        )

    def create_new_df_from_labels(self, column_names, rows, add_timestamp=True):
        """
        Create a new DataFrame with the specified column names.

        Args:
            column_names (list): List of column names for the DataFrame.
            rows (list): List of dictionaries, where each dictionary represents a row of data.
        """
        if add_timestamp:
            column_names.append('last_update')
            for row in rows:
                row['last_update'] = datetime.now()
            self.add_timestamp = True
        else:
            self.add_timestamp = False
        self.df = pd.DataFrame(columns=column_names)
        self.df = pd.concat([self.df, pd.DataFrame(rows)], ignore_index=True)

        print("Creating or updating new csv at path:", self.filename)
        self.df.to_csv(self.filename, index=False)

    def get_value(self, row_name, column_name):
        """
        Get the value from the DataFrame for a specific row and column.

        Args:
            row_name (str): The name of the row to look up.
            column_name (str): The name of the column to look up.

        Returns:
            The value from the DataFrame at the specified row and column.
        """
        return self.df.loc[
            self.df[self.indexing_row_name] == row_name, column_name
        ].values[0]

    def update_value(self, row_name, column_name, value, save_to_file=True):
        """
        Update the value in the DataFrame for a specific row and column and saves to the csv file if requested.

        Args:
            row_name (str): The name of the row to update.
            column_name (str): The name of the column to update.
            value: The new value to set in the DataFrame.
        """
        self.df.loc[self.df[self.indexing_row_name] == row_name, column_name] = value
        if self.add_timestamp:
            self.df.loc[self.df[self.indexing_row_name] == row_name, 'last_update'] = (
                datetime.now()
            )
        if save_to_file:
            print("Creating or updating new csv at path:", self.filename)
            self.df.to_csv(self.filename, index=False)

    def get_last_update(self, row_name):
        if self.add_timestamp:
            return self.get_value(row_name, 'last_update')
        return None

    def update_last_update(self, stor_name, save_to_file=True):
        if not self.add_timestamp:
            return
        self.update_value(
            stor_name, 'last_update', datetime.now(), save_to_file=save_to_file
        )

    def get_columns(self):
        return self.df.columns.tolist()

    def get_all(self, row_name):
        return self.df[self.df[self.indexing_row_name] == row_name].values[0]

    def update_all(self, *args, save_to_file=True):
        """
        Update all values in the DataFrame for a specific row and always saves the file.
        Args:
            *args: A tuple containing the values to update in the order of the columns (excluding the timestamp).
                   The first value should be the row name, followed by values for each column.
            save_to_file (bool): Whether to save the updated DataFrame to the CSV file.
        """
        num_cols = len(self.get_columns())
        if self.add_timestamp:
            num_cols -= 1  # Exclude the timestamp column for check
        if len(args) != num_cols:
            raise ValueError(
                f"Number of arguments must match the columns in the DataFrame: {self.df.columns.tolist()}"
            )
        for i in range(1, num_cols):
            column = self.get_columns()[i]
            self.update_value(args[0], column, args[i], save_to_file=False)
        self.update_last_update(args[0], save_to_file=save_to_file)

    def append_dataset(self, *args, add_timestamp=True, save_to_file=True):
        """
        Append a new row to the DataFrame with the provided values.
        Args:
            *args: A tuple containing the values to append in the order of the columns.
                   The first value should be the row_name, followed by values for each column, excluding the time stamp which will be added automatically.
        """
        if len(args) != len(self.df.columns):
            raise ValueError(
                f"Number of arguments must match the columns in the DataFrame: {self.df.columns.tolist()}"
            )
        new_row = dict()
        for i, column in enumerate(self.df.columns):
            if (
                i == len(self.df.columns.tolist()) - 1 and add_timestamp
            ):  # last column is timestamp
                new_row[column] = datetime.now()
            else:
                new_row[column] = args[i]
        self.df = self.df.append(new_row, ignore_index=True)
        if save_to_file:
            print("Creating or updating new csv at path:", self.filename)
            self.df.to_csv(self.filename, index=False)

    # check whether the data is up-to-date
    def is_up_to_date(self, row_name, max_time_diff=7200):
        last_update = self.get_last_update(row_name)
        # Define the format of the date string
        date_format = "%Y-%m-%d %H:%M:%S.%f"
        # Convert the string to a datetime object
        last_update_object = datetime.strptime(last_update, date_format)
        time_diff = (datetime.now() - last_update_object).total_seconds()
        return time_diff < max_time_diff

    def create_copy(self, new_filename=None):
        expts_path = ''
        # print(f"expts_path: {expts_path}")

        if new_filename is None:
            name, ext = os.path.splitext(os.path.basename(self.filename))
            new_filename = os.path.join(expts_path, f"{name}_copy{ext}")
        else:
            new_filename = os.path.join(expts_path, new_filename)
        print(f"Creating a copy of the dataset at path: {new_filename}")
        self.df.to_csv(new_filename, index=False)
        return new_filename

    def compare_with(self, other_dataset):
        """
        Compare the current dataset with another dataset and identify differences.

        This method iterates through the rows of the current dataset (`self.df`) and compares
        each row with the corresponding row in the `other_dataset` based on the indexing_row_name column.
        It identifies differences in column values between the two datasets and returns a list
        of discrepancies.

        Args:
            other_dataset: An instance of the same dataset class containing a DataFrame (`df`)
                   to compare against.

        Returns:
            list: A list of dictionaries, where each dictionary represents a difference. Each
              dictionary contains the following keys:
              - whatever indexing_row_name is: The identifier of the row being compared.
              - 'column': The column where the difference was found, or 'all' if the row
                      exists in `self` but is missing in `other_dataset`.
              - 'self_value': The value in the current dataset.
              - 'other_value': The value in the other dataset, or 'missing' if the row
                       does not exist in `other_dataset`.

        Example:
            differences = dataset1.compare_with(dataset2)
            for diff in differences:
            print(diff)
        """
        differences = []
        for _, row in self.df.iterrows():
            row_name = row[self.indexing_row_name]
            if row_name in other_dataset.df[self.indexing_row_name].values:
                other_row = other_dataset.df[
                    other_dataset.df[self.indexing_row_name] == row_name
                ].iloc[0]
                for column in self.df.columns:
                    if row[column] != other_row[column]:
                        differences.append(
                            {
                                self.indexing_row_name: row_name,
                                'column': column,
                                'self_value': row[column],
                                'other_value': other_row[column],
                            }
                        )
            else:
                differences.append(
                    {
                        self.indexing_row_name: row_name,
                        'column': 'all',
                        'self_value': 'exists',
                        'other_value': 'missing',
                    }
                )
        return differences

    def save_to_file(self, filepath):
        """
        Save the current dataset to a specified file.

        Args:
            filepath (str): The path to the file where the dataset should be saved.
        """
        print("Creating or updating new csv at path:", filepath)
        self.df.to_csv(filepath, index=False)


class StorageManSwapDataset(MMDataset):
    def __init__(self, filename='man1_storage_swap_dataset.csv'):
        super().__init__(filename=filename)

    def create_new_df(self):
        column_names = [
            'stor_name',
            'freq (MHz)',
            'precision (MHz)',
            'pi (mus)',
            'h_pi (mus)',
            'gain (DAC units)',
        ]

        rows = []
        for idx in range(1, 13, 1):
            row = {
                'stor_name': 'M1-S' + str(idx),
                'freq (MHz)': -1,
                'precision (MHz)': -1,
                'pi (mus)': -1,
                'h_pi (mus)': -1,
                'gain (DAC units)': -1,
            }
            rows.append(row)

        # also add for the manipulate
        row = {
            'stor_name': 'M1',
            'freq (MHz)': -1,
            'precision (MHz)': -1,
            'pi (mus)': -1,
            'h_pi (mus)': -1,
            'gain (DAC units)': -1,
        }
        rows.append(row)
        self.create_new_df_from_labels(column_names, rows, add_timestamp=True)

    # fetch the data from the csv file
    def get_freq(self, stor_name):
        return self.get_value(stor_name, 'freq (MHz)')

    def get_precision(self, stor_name):
        return self.get_value(stor_name, 'precision (MHz)')

    def get_pi(self, stor_name):
        return self.get_value(stor_name, 'pi (mus)')

    def get_h_pi(self, stor_name):
        return self.get_value(stor_name, 'h_pi (mus)')

    def get_gain(self, stor_name):
        self.df['gain (DAC units)'] = self.df['gain (DAC units)'].astype(int)
        return self.get_value(stor_name, 'gain (DAC units)')

    # update the data in the csv file
    def update_freq(self, stor_name, freq):
        self.update_value(stor_name, 'freq (MHz)', freq)

    def update_precision(self, stor_name, precision):
        self.update_value(stor_name, 'precision (MHz)', precision)

    def update_pi(self, stor_name, pi):
        self.update_value(stor_name, 'pi (mus)', pi)

    def update_h_pi(self, stor_name, h_pi):
        self.update_value(stor_name, 'h_pi (mus)', h_pi)

    def update_gain(self, stor_name, gain):
        self.update_value(stor_name, 'gain (DAC units)', gain)
        self.df['gain (DAC units)'] = self.df['gain (DAC units)'].astype(int)


class FloquetStorageSwapDataset(MMDataset):
    def __init__(self, filename='floquet_storage_swap_dataset.csv'):
        super().__init__(filename=filename)

    def create_new_df(self):
        column_names = [
            'stor_name',
            'pi_frac',  # this pulse implements a pi / pi_frac pulse
            'freq (MHz)',
            'gain (DAC units)',
            'len (mus)',
            'ramp_sigma (mus)',
        ]
        for idx in range(1, 8, 1):
            column_names.append(
                'phase_from_M1-S' + str(idx) + ' (deg)'
            )  # phase of the pulse on M1-Sx

        rows = []
        for idx in range(1, 8, 1):
            row = {
                'stor_name': 'M1-S' + str(idx),
                'pi_frac': -1,
                'freq (MHz)': -1.0,
                'gain (DAC units)': -1,
                'len (mus)': -1.0,
                'ramp_sigma (mus)': -1,
            }
            for i in range(1, 8, 1):
                row['phase_from_M1-S' + str(i) + ' (deg)'] = 0.0
            rows.append(row)

        self.create_new_df_from_labels(column_names, rows, add_timestamp=True)

    # fetch the data from the csv file
    def get_freq(self, stor_name):
        return self.get_value(stor_name, 'freq (MHz)')

    def get_pi_frac(self, stor_name):
        return self.get_value(stor_name, 'pi_frac')

    def get_len(self, stor_name):
        return self.get_value(stor_name, 'len (mus)')

    def get_gain(self, stor_name):
        self.df['gain (DAC units)'] = self.df['gain (DAC units)'].astype(int)
        return self.get_value(stor_name, 'gain (DAC units)')

    def get_ramp_sigma(self, stor_name):
        return self.get_value(stor_name, 'ramp_sigma (mus)')

    def get_phase_from(self, stor_name, from_stor_name):
        return self.get_value(stor_name, f'phase_from_{from_stor_name} (deg)')

    # update the data in the csv file
    def update_freq(self, stor_name, freq):
        self.update_value(stor_name, 'freq (MHz)', freq)

    def update_pi_frac(self, stor_name, pi_frac):
        self.update_value(stor_name, 'pi_frac', pi_frac)

    def update_len(self, stor_name, length):
        self.update_value(stor_name, 'len (mus)', length)

    def update_gain(self, stor_name, gain):
        self.update_value(stor_name, 'gain (DAC units)', gain)
        self.df['gain (DAC units)'] = self.df['gain (DAC units)'].astype(int)

    def update_ramp_sigma(self, stor_name, ramp_sigma):
        self.update_value(stor_name, 'ramp_sigma (mus)', ramp_sigma)

    def update_phase_from(self, stor_name, from_stor_name, phase):
        self.update_value(stor_name, f'phase_from_{from_stor_name} (deg)', phase)

    def import_from_swap_dataset(
        self, stor_man_ds: StorageManSwapDataset, gain_div: int, pi_div: int
    ):
        """
        Import the frequency, gain and pulse length from the pi/hpi swap dataset
        Freq is directly copied for all 7 modes
        gain is copied and divided by gain_div
        length is the pi pulse length divided by pi_div
        pi_frac is the product of the gain and pi divisions
        """
        for stor_name in range(1, 8):
            stor_name_str = 'M1-S' + str(stor_name)
            orig_freq = stor_man_ds.get_freq(stor_name_str)
            orig_gain = stor_man_ds.get_gain(stor_name_str)
            orig_pi = stor_man_ds.get_pi(stor_name_str)
            self.update_freq(stor_name_str, orig_freq)
            self.update_gain(stor_name_str, orig_gain // gain_div)
            self.update_len(stor_name_str, orig_pi / pi_div)
            self.update_pi_frac(stor_name_str, gain_div * pi_div)

