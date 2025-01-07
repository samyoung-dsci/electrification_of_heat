#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
import qa_functions as qa
import utils
import temperature_data_fns as td
import numpy as np
import matplotlib.pyplot as plt
import cleaning_fns as clf
from tqdm import tqdm
import plotly.io as pio
from IPython.display import Image
pio.orca.config.use_xvfb = True
pio.orca.config.save()

pd.set_option("display.max_columns", None)


# In[ ]:


# Set file format for the raw data you have, either parquet or csv
file_format = qa.set_file_format(file_format="csv")


# In[ ]:


# Specify which folder contains your local directory
eoh_folder = os.environ["EoH"]

location, location_in_raw, location_out, location_out_cleaned = utils.create_folder_structure(os.path.join(eoh_folder))

# Generate a list of all files to be run through the code
files_info = []

for home in os.listdir(location_in_raw):
    size = os.path.getsize(os.path.join(location_in_raw, home))
    file_info = [home, round(size / 1024)]
    files_info.append(file_info)
# Remove properties which do not have more than 1KB of data
files_info = pd.DataFrame(files_info, columns=["Property_Name", "Size_KB"])
files_info = files_info[files_info["Size_KB"] > 1]
files_list = list(files_info.iloc[:, 0])
files_list = [file.split(".")[0] for file in files_list]


# In[ ]:


# Define the performance factor ranges for different time scales
# For short time scales, we expect higher variation in the performance factor
# short time periods are around a day, long are around a year, medium is in between these two
spf_ranges = {
    "short": {"min": 0.75, "max": 7.5},
    "medium": {"min": 0.9, "max": 6.5},
    "long": {"min": 1.5, "max": 5.0},
}


# In[ ]:


# This code is commented out as it takes some time to run
# You can run the code in this cell to generate the temperature_stats.csv
# Alternately if you have already generated the csv then
# you can load the csv in the following cell to view the plots below

all_stats = pd.DataFrame()
for file in tqdm(files_list):
    temperature_data = utils.load_temperature_data(os.path.join(location_in_raw, file), file_format=file_format)
    temperature_data = qa.round_timestamps(temperature_data, n_mins=2)
    path = os.path.join(location_out, "plots", "temperature", file + ".png")
    td.plot_data(temperature_data, path=path)
    temperature_data = temperature_data.reset_index()
    stats = td.get_temperature_stats(temperature_data, file)
    all_stats = pd.concat([all_stats, stats])

path = os.path.join(location_out, "temperature_stats.csv")
all_stats.to_csv(path, index=False)


# In[ ]:


# Using the temperature_stats.csv file, we create classifiers and make predictions
# and create the homes_dropped subset of homes
(
    classifier_A,
    classifier_B,
    predictions_A,
    predictions_B,
    stats_to_use,
    all_stats,
    homes_dropped,
    graph_classififier_A,
    graph_classififier_B,
) = clf.classify_and_predict(location_out, files_list)


# In[ ]:


Image(graph_classififier_A.create_png())


# In[ ]:


Image(graph_classififier_B.create_png())


# In[ ]:


# The code in this cell generates 3 more subsets of homes:
# unflagged homes, single_flagged_homes, double_flagged homes

output = stats_to_use.copy()
output["prediction_A"] = predictions_A
output["prediction_B"] = predictions_B
output["flagged_A"] = output["prediction_A"] != (output["sensor_type"] == "Hot_Water_Flow_Temperature")
output["flagged_B"] = output["prediction_B"] != (output["sensor_type"] == "Hot_Water_Flow_Temperature")

# There is a fair difference between how the splits behave, we will have a look at all of them
output["flagged"] = output["flagged_A"] | output["flagged_B"]

home_flags = output.groupby(["Property_ID"])["flagged"].sum()
output = pd.merge(output, home_flags, on="Property_ID", how="left", suffixes=[None, "_per_home"])

# Merge in all_stats to get the issue for the homes dropped for having < 50% temp data
output = pd.merge(
    output,
    all_stats[["sensor_type", "Property_ID", "issue"]],
    on=["sensor_type", "Property_ID"],
    how="outer",
)

flagged_homes = output[output["flagged_per_home"] > 0]
flagged_homes = flagged_homes.groupby("Property_ID").agg({"flagged_per_home": "max"})
unflagged_homes = np.sort(output.loc[output["flagged_per_home"] == 0]["Property_ID"].unique())
single_flagged_homes = flagged_homes.loc[flagged_homes["flagged_per_home"] == 1]
double_flagged_homes = flagged_homes.loc[flagged_homes["flagged_per_home"] == 2]
single_flagged_homes = single_flagged_homes.index.sort_values()
double_flagged_homes = double_flagged_homes.index.sort_values()


# In[ ]:


# This cell is where cleaning for each home takes

home_classifications = {
    "unflagged": unflagged_homes,
    "single_flagged": single_flagged_homes,
    "double_flagged": double_flagged_homes,
    "dropped": homes_dropped,
}

# Loop through each home
all_dropped_dupes = 0
window_method = "best"
homes_with_error = []
home_summary = []
all_windows = pd.DataFrame()
all_homes_alteration_record = pd.DataFrame()
alteration_record = pd.DataFrame()

for home_classification in ["unflagged", "single_flagged", "double_flagged", "dropped"]:
    for home in tqdm(home_classifications.get(home_classification)):
        
        temp_cleaning_type = home_classification

        home_summary_part, alteration_record, cleaned_data = clf.cleaning(
            home,
            location,
            location_in_raw,
            location_out_cleaned,
            spf_ranges,
            all_dropped_dupes,
            alteration_record,
            all_homes_alteration_record,
            temp_cleaning_type,
            output,
            classifier_A,
            classifier_B,
            file_format,
        )

        home_summary.append(home_summary_part)
        all_homes_alteration_record = pd.concat([all_homes_alteration_record, alteration_record])

home_summary = pd.concat(home_summary)
home_summary


# In[ ]:


# Save the outcomes output
output = output.sort_values("Property_ID")
output.to_csv(os.path.join(location_out, "temperature_stats_with_outcome.csv"))


# In[ ]:


# Save home summary
home_summary.to_csv(os.path.join(location_out, "home_summary_partial_1.csv"), index=False)


# In[ ]:


print(f"Total dropped duplicates: {all_dropped_dupes}")

# Save out the alteration record for all the homes
all_homes_alteration_record.to_csv(os.path.join(location_out, "summary_of_alterations.csv"), index=False)

