#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
from tqdm import tqdm
import qa_functions as qa
import numpy as np

# Set no limit to the number of columns displayed
pd.set_option("display.max_columns", None)


# In[ ]:


# Specify location of specific folders
location = os.environ["EoH"]

location_out = os.path.join(location, "processed")
location_out_cleaned = os.path.join(location_out, "cleaned")
location_out_cleaned_half_hourly = os.path.join(location_out_cleaned, "cleaned_half_hourly")
location_out_cleaned_half_hourly_in_window = os.path.join(location_out_cleaned, "cleaned_half_hourly_in_window")
location_out_cleaned_daily = os.path.join(location_out_cleaned, "cleaned_daily")
location_out_cleaned_daily_in_window = os.path.join(location_out_cleaned, "cleaned_daily_in_window")
location_out_raw = os.path.join(location_out, "raw")
location_out_raw_daily = os.path.join(location_out_raw, "raw_daily")
location_out_raw_daily_in_window = os.path.join(location_out_raw, "raw_daily_in_window")
location_out_for_publication = os.path.join(location_out, "for_publication")

# Create folders if they do not exist
if not os.path.exists(location_out_cleaned_half_hourly):
    os.mkdir(location_out_cleaned_half_hourly)
if not os.path.exists(location_out_cleaned_half_hourly_in_window):
    os.mkdir(location_out_cleaned_half_hourly_in_window)
if not os.path.exists(location_out_cleaned_daily):
    os.mkdir(location_out_cleaned_daily)
if not os.path.exists(location_out_cleaned_daily_in_window):
    os.mkdir(location_out_cleaned_daily_in_window)
if not os.path.exists(location_out_raw):
    os.mkdir(location_out_raw)
if not os.path.exists(location_out_raw_daily):
    os.mkdir(location_out_raw_daily)
if not os.path.exists(location_out_raw_daily_in_window):
    os.mkdir(location_out_raw_daily_in_window)
if not os.path.exists(location_out_for_publication):
    os.mkdir(location_out_for_publication)

# Generate a list of all files to be run through the code
files_info = []

# Pull info size of files
for home in os.listdir(os.path.join(location_out_cleaned, "cleaned")):
    size = os.path.getsize(os.path.join(location_out_cleaned, "cleaned", home))
    file_info = [home, round(size / 1024)]
    files_info.append(file_info)
# Remove properties which do not have more than 1KB of data
files_info = pd.DataFrame(files_info, columns=["Property_Name", "Size_KB"])
files_info = files_info[files_info["Size_KB"] > 1]
files_list = list(files_info.iloc[:, 0])
files_list = [file.split(".")[0] for file in files_list]


# In[ ]:


# We use this file to create a dataset which includes only homes with a valid window, i.e. are used in the analysis
home_summary = pd.read_csv(os.path.join(location_out, "home_summary_with_install_info.csv"))


# In[ ]:


# Loop through different time resolutions and create datasets

for home in tqdm(files_list):
    df = pd.read_parquet(os.path.join(location_out_cleaned, rf"cleaned/{home}.parquet"))
    df = df.pivot_table(index="Timestamp", columns="sensor_type", values="value")
    energy_cols = df.columns[df.columns.str.contains("Energy")]
    temp_cols = df.columns[df.columns.str.contains("Temp")]
    for time_res in ["30min", "1d"]:
        # resample
        df_energy_resampled = df[energy_cols].resample(f"{time_res}").max()
        df_temp_resampled = df[temp_cols].resample(f"{time_res}").mean()
        df_resampled = df_energy_resampled.merge(df_temp_resampled, left_index=True, right_index=True)
        df_resampled = df_resampled.reset_index().rename_axis("", axis=1)
        property_id = home[-7:]
        df_resampled["Property_ID"] = property_id
        if time_res == "30min":
            df_resampled["half-hour"] = df_resampled["Timestamp"].dt.round("30min").dt.time
            df_resampled.to_parquet(
                os.path.join(location_out_cleaned, "cleaned_half_hourly", f"Property_ID={property_id}.parquet")
            )
            time_res_column = "Timestamp"
            window_save_location = location_out_cleaned_half_hourly_in_window
        elif time_res == "1d":
            df_resampled = df_resampled.rename({"Timestamp": "Date"}, axis=1)
            df_resampled.to_parquet(os.path.join(location_out_cleaned, "cleaned_daily", f"Property_ID={property_id}.parquet"))
            time_res_column = "Date"
            window_save_location = location_out_cleaned_daily_in_window
        # Filter datasets for window start/end range and then save
        try:
            if home_summary[home_summary["Property_ID"] == property_id]["Removed_from_analysis"].values[0] == False:
                df_resampled = df_resampled[
                    ~(
                        (
                            (
                                df_resampled[time_res_column]
                                < home_summary[home_summary["Property_ID"] == property_id]["window_start"].values[0]
                            )
                            | (
                                df_resampled[time_res_column]
                                >= home_summary[home_summary["Property_ID"] == property_id]["window_end"].values[0]
                            )
                        )
                    )
                ]
                df_resampled = df_resampled.sort_values([time_res_column])
                df_resampled.to_parquet(os.path.join(window_save_location, f"Property_ID={property_id}.parquet"))
        except:
            print(f"{home} did not have an in-window dataset created for it, likely does not have a valid window")


# In[ ]:


# Set format of raw data
raw_file_format = "csv"


# In[ ]:


# Create daily versions of raw data

for home in tqdm(files_list):
    if raw_file_format == "csv":
        df = pd.read_csv(os.path.join(location, rf"raw/{home}.csv")).rename({"Timestamp": "timestamp"}, axis=1).set_index("timestamp")
    elif raw_file_format == "parquet":
        df = pd.read_parquet(os.path.join(location, rf"raw/{home}.parquet"))        
    df = df.pivot_table(index=df.index, columns="sensor_type", values="value")
    df.index = pd.to_datetime(df.index)
    energy_cols = df.columns[df.columns.str.contains("Energy")]
    temp_cols = df.columns[df.columns.str.contains("Temp")]
    time_res = "1d"
    # resample
    df_energy_resampled = df[energy_cols].resample(f"{time_res}").max()
    df_temp_resampled = df[temp_cols].resample(f"{time_res}").mean()
    df_resampled = df_energy_resampled.merge(df_temp_resampled, left_index=True, right_index=True)
    df_resampled = df.resample(f"1d").max()
    df_resampled = df_resampled.reset_index().rename_axis("", axis=1)
    property_id = home[[-7:]]
    df_resampled["Property_ID"] = property_id
    df_resampled = df_resampled.rename({"timestamp": "Date"}, axis=1)
    df_resampled.to_parquet(os.path.join(location_out_raw_daily, f"{property_id}.parquet"))
    # Filter datasets for window start/end range and then save
    try:
        if home_summary[home_summary["Property_ID"] == property_id]["Removed_from_analysis"].values[0] == False:
            df_resampled = df_resampled[
                ~(
                    (
                        (
                            df_resampled["Date"]
                            < home_summary[home_summary["Property_ID"] == property_id]["window_start"].values[0]
                        )
                        | (
                            df_resampled["Date"]
                            >= home_summary[home_summary["Property_ID"] == property_id]["window_end"].values[0]
                        )
                    )
                )
            ]
            df_resampled = df_resampled.sort_values(["Date"])
            df_resampled.to_parquet(os.path.join(location_out_raw_daily_in_window, f"Property_ID={property_id}.parquet"))
    except:
        print(f"{home} did not have an in-window dataset created for it, likely does not have a valid window")


# In[ ]:


# Combine parquet files into a single csv file

for time_res in ["half_hourly", "daily"]:
    print(time_res)
    dfs = pd.DataFrame()
    for file in os.listdir(eval(f"location_out_cleaned_{time_res}")):
        df = pd.read_parquet(os.path.join(location_out_cleaned, f"cleaned_{time_res}", file))
        
        energy_cols = df.columns[df.columns.str.contains("Energy")]
        temp_cols = df.columns[df.columns.str.contains("Temp")]

        # Input columns are not in an ideal order, so we re-order them here
        if time_res == "half_hourly":
            df = pd.concat([df[["Property_ID", "Timestamp", "half-hour"]], round(df[energy_cols], 3), round(df[temp_cols], 2)], axis=1).sort_values(["Property_ID", "Timestamp"])
        elif time_res == "daily":
            df = pd.concat([df[["Property_ID", "Date"]], round(df[energy_cols], 3), round(df[temp_cols], 2)], axis=1).sort_values(["Property_ID", "Date"])
        dfs = pd.concat([dfs, df])
    dfs.to_csv(os.path.join(location_out_for_publication, f"cleaned_{time_res}.csv"), index=False)

