#!/usr/bin/env python
# coding: utf-8

# ### COP/Energy Output vs Flow/External Temperature

# To determine the impact of large-scale rollout of heat pumps on the wider energy system, it is 
# necessary to consider the variation in COP (and therefore power draw) of heat pumps over 
# different 30-minute periods. It is also necessary to consider how these factors vary during the 
# coldest periods of the year. The following subsections consider the impact of varying flow 
# temperatures and external temperatures on COP both by considering the mean flow and 
# external temperature over 30-minute periods and by considering the COP over the coldest days 
# of the year. 

# In[ ]:


import pandas as pd
import chart_functions as cf
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import os
import qa_functions as qa
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

pio.renderers.default = "notebook"

# Set no limit to the number of columns displayed
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 120)

# Set plotly colour scheme
pio.templates["EoH"] = go.layout.Template(
    layout_colorway=["#%02x%02x%02x" % rgb_colour for rgb_colour in cf.colour_scheme().values()]
)
pio.templates.default = "EoH"

# Set marker size and opacity
opacity = 0.3
marker_size = 2

pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


# Specify location for read and write
location = os.environ["EoH"]


# In[ ]:


# If the all_homes.parquet file is not already generated, the following code should be run
# If the file is already generated then move on to the next code block
# Read in data, 1 row per home
homes_output = pd.read_csv(os.path.join(location, "processed", "home_summary_with_install_info.csv"))

# Read in the 30-minutely dataset using schema to ensure we get all columns and not just the columns present in the first file

dataset_dir = os.path.join(location, "processed", "cleaned", "cleaned_half_hourly_in_window")
dataset = ds.dataset(dataset_dir)
schemas = [pq.read_schema(dataset_file) for dataset_file in dataset.files]

unified_schema = pa.unify_schemas(schemas)

dataset = ds.dataset(dataset_dir, schema=unified_schema)
df_all = dataset.to_table().to_pandas()

df_all.to_parquet(os.path.join(location, "processed", "cleaned", "cleaned_half_hourly_in_window_single_file", "all_homes.parquet"))


# In[107]:


# If the file is generated it can be loaded here instead
df_all = pd.read_parquet(os.path.join(location, "processed", "cleaned", "cleaned_half_hourly_in_window_single_file", "all_homes.parquet"))

# Define energy and temp cols
energy_cols = df_all.columns[df_all.columns.str.contains("Energy")].to_list()
temp_cols = df_all.columns[df_all.columns.str.contains("Temp")].to_list()

# Load auxillary data
home_summary = pd.read_csv(os.path.join(location, "processed", "home_summary_with_install_info.csv"))

# Filter only for homes that were included in the analysis (ensures data quality), may want to look at just the data quality column instead
home_summary = home_summary[home_summary["Removed_from_analysis"] == False]

# Choose cols to use from home_summary
id_cols = ["Property_ID", "HP_Refrigerant", "HP_Size_kW", "HP_Type", "HP_Type_2", "Delivery_Contractor"]

# Merge dataframes
df_all = df_all.merge(home_summary[id_cols], on="Property_ID")


# In[108]:


# Set resolution for resampling to 30 minutes
resolution = "30min"

df_all_energy = df_all.set_index("Timestamp")[energy_cols + id_cols].groupby(id_cols)[energy_cols].resample(resolution).max(numeric_only=True).reset_index()
df_all_temp = df_all.set_index("Timestamp")[temp_cols + id_cols].groupby(id_cols)[temp_cols].resample(resolution).mean(numeric_only=True).reset_index()
df_all_resampled = df_all_energy.merge(df_all_temp, on=id_cols+["Timestamp"], copy=False)

""" 
Remove cumulative

By using diff(), we are taking the difference between values on each row. As the data can contain gaps, this can sometimes lead
to large values being assigned to energy columns. As we have a filter to remove COP values above 8 and also only take a median of
a large number of energy values, the analysis is not affected by potential high anomalies within the data.
"""

df_all_resampled = df_all_resampled.sort_values(["Property_ID", "Timestamp"])
df_all_resampled[energy_cols] = df_all_resampled.groupby("Property_ID")[energy_cols].diff().fillna(0)

# replace nans with 0

df_all_resampled[
    ["Circulation_Pump_Energy_Consumed", "Immersion_Heater_Energy_Consumed", "Back-up_Heater_Energy_Consumed"]
] = df_all_resampled[
    ["Circulation_Pump_Energy_Consumed", "Immersion_Heater_Energy_Consumed", "Back-up_Heater_Energy_Consumed"]
].fillna(
    0
)

# Add additional columns
df_all_resampled["COPH2"] = qa.spf(df_all_resampled, metric="spfh2")
df_all_resampled["COPH4"] = qa.spf(df_all_resampled, metric="spfh4")
df_all_resampled["date"] = df_all_resampled["Timestamp"].dt.date
df_all_resampled["Month_name"] = df_all_resampled["Timestamp"].dt.month_name()
df_all_resampled["Month"] = df_all_resampled["Timestamp"].dt.month
df_all_resampled["Year"] = df_all_resampled["Timestamp"].dt.year

df_all_resampled["Rounded_External_Air_Temperature"] = round(df_all_resampled["External_Air_Temperature"])
df_all_resampled["Rounded_Flow_Temperature"] = round(df_all_resampled["Heat_Pump_Heating_Flow_Temperature"]) 
df_all_resampled["hp_flow-return"] = df_all_resampled["Heat_Pump_Heating_Flow_Temperature"] - df_all_resampled["Heat_Pump_Return_Temperature"]
df_all_resampled["hp_flow-ext_temp"] = df_all_resampled["Heat_Pump_Heating_Flow_Temperature"] - df_all_resampled["External_Air_Temperature"]

df_all_resampled.head()


# In[109]:


def temp_vs_cop_calcs(df: pd.DataFrame, show_plots: bool, min_ext_air_temp: int, max_ext_air_temp: int, color_by=None):

    """This function takes uses the input dataframe to create two summary tables and plots which are filtered between acceptable ranges for
    external air temperature, heat pump energy output, flow temperature.

    Args:
        df (pd.DataFrame): dataframe containing data on external air temperature, heat pump energy output and flow temperature
        show_plots (bool): choose whether to show plots or not
        min_ext_air_temp: choose the minimum external air temperature in the summary table and charts
        max_ext_air_temp: choose the maximum external air temperature in the summary table and charts
        color_by: defaults to None, choose a parameter to split the data into coloured groups.

    Returns:
        df_all_resampled_ext_temp_stats (pd.DataFrame): _description_
        df_all_resampled_flow_temp_stats (pd.DataFrame): 
    """

    df_all_resampled_ext_temp_stats = df[(df["Rounded_External_Air_Temperature"] >= min_ext_air_temp) & (df["Rounded_External_Air_Temperature"] <= max_ext_air_temp)]
    df_all_resampled_ext_temp_stats = df_all_resampled_ext_temp_stats[(df_all_resampled_ext_temp_stats["Heat_Pump_Energy_Output"] > 0) & (df_all_resampled_ext_temp_stats["Whole_System_Energy_Consumed"] > 0)]
    df_all_resampled_ext_temp_stats["Rounded_External_Air_Temperature"] = df_all_resampled_ext_temp_stats["Rounded_External_Air_Temperature"].astype(int)
    if color_by == None:
        df_all_resampled_ext_temp_stats = df_all_resampled_ext_temp_stats.groupby("Rounded_External_Air_Temperature").agg({"COPH4": ["count", "median"], "Heat_Pump_Energy_Output": ["median", "max"], "Whole_System_Energy_Consumed": ["median", "max"]})
    else:
        df_all_resampled_ext_temp_stats = df_all_resampled_ext_temp_stats.groupby([color_by, "Rounded_External_Air_Temperature"]).agg({"COPH4": ["count", "median"], "Heat_Pump_Energy_Output": ["median", "max"], "Whole_System_Energy_Consumed": ["median", "max"]}).reset_index().set_index("Rounded_External_Air_Temperature")
    df_all_resampled_ext_temp_stats = round(df_all_resampled_ext_temp_stats, 2)

    df_all_resampled_flow_temp_stats = df[df["Rounded_Flow_Temperature"] >= 30]
    df_all_resampled_flow_temp_stats = df_all_resampled_flow_temp_stats[(df_all_resampled_flow_temp_stats["Heat_Pump_Energy_Output"] > 0) & (df_all_resampled_flow_temp_stats["Whole_System_Energy_Consumed"] > 0)]
    df_all_resampled_flow_temp_stats["Rounded_Flow_Temperature"] = df_all_resampled_flow_temp_stats["Rounded_Flow_Temperature"].astype(int)
    df_all_resampled_flow_temp_stats["Rounded_Flow_Temperature"] = np.where(df_all_resampled_flow_temp_stats["Rounded_Flow_Temperature"] >= 65, ">=65", df_all_resampled_flow_temp_stats["Rounded_Flow_Temperature"])
    if color_by == None:
        df_all_resampled_flow_temp_stats = df_all_resampled_flow_temp_stats.groupby("Rounded_Flow_Temperature").agg({"COPH4": ["count", "median"], "Heat_Pump_Energy_Output": ["median", "max"], "Whole_System_Energy_Consumed": ["median", "max"]})
    else:
        df_all_resampled_flow_temp_stats = df_all_resampled_flow_temp_stats.groupby([color_by, "Rounded_Flow_Temperature"]).agg({"COPH4": ["count", "median"], "Heat_Pump_Energy_Output": ["median", "max"], "Whole_System_Energy_Consumed": ["median", "max"]}).reset_index().set_index("Rounded_Flow_Temperature")
    df_all_resampled_flow_temp_stats = round(df_all_resampled_flow_temp_stats, 2)

    if show_plots:
        fig = px.line(df_all_resampled_ext_temp_stats, x=df_all_resampled_ext_temp_stats.index, y=df_all_resampled_ext_temp_stats["COPH4"]["median"].values, color=color_by, width=800)
        fig.update_xaxes(title="External temperature °C")
        fig.update_yaxes(title="COPH4")
        fig.show()

        fig = px.bar(df_all_resampled_ext_temp_stats, x=df_all_resampled_ext_temp_stats.index, y=df_all_resampled_ext_temp_stats["COPH4"]["count"].values, color=color_by, width=800, height=200)
        fig.update_yaxes(showticklabels=False, title="Data points")
        fig.update_xaxes(title="External temperature °C")
        fig.update_layout(bargap=0, showlegend=False)
        fig.show()

        fig = px.line(df_all_resampled_ext_temp_stats, x=df_all_resampled_ext_temp_stats.index, y=df_all_resampled_ext_temp_stats["Whole_System_Energy_Consumed"]["median"].values, color=color_by, width=800)
        fig.update_xaxes(title="External temperature °C")
        fig.update_yaxes(title="Whole_System_Energy_Consumed")
        fig.show()

        fig = px.line(df_all_resampled_flow_temp_stats, x=df_all_resampled_flow_temp_stats.index, y=df_all_resampled_flow_temp_stats["COPH4"]["median"].values, color=color_by, width=800)
        fig.update_xaxes(title="Flow temperature °C")
        fig.update_yaxes(title="COPH4")
        fig.show()

        fig = px.bar(df_all_resampled_flow_temp_stats, x=df_all_resampled_flow_temp_stats.index, y=df_all_resampled_flow_temp_stats["COPH4"]["count"].values, color=color_by, width=800, height=200)
        fig.update_yaxes(showticklabels=False, title="Data points")
        fig.update_xaxes(title="Flow temperature °C")
        fig.update_layout(bargap=0, showlegend=False)
        fig.show()

    return df_all_resampled_ext_temp_stats, df_all_resampled_flow_temp_stats


# ASHPs only

# The first 2 graphs show a significant increase in COP as the external temperature increased up 10°C 
# (COP(H4) = 3.37) when many of the heat pumps were using more energy for hot water 
# production than heating. As the external temperature after this point rose, the COP decreased
# but it should be noted that, due to switching from heating to hot water mode, the energy 
# consumed after this point also decreased. The sample size at the lower extreme of the external 
# temperatures is very small however, the decrease in COP (alongside external temperature) 
# continued to this point where the COP was 2.08 as the external temperature decreased to -
# 10°C.

# In[110]:


df_ashps = df_all_resampled[df_all_resampled["HP_Type_2"] == "ASHPs"]
df_ext_temp, df_flow_temp = temp_vs_cop_calcs(df_ashps, show_plots=True, min_ext_air_temp = -10, max_ext_air_temp=18)


# In[111]:


df_ext_temp


# In[112]:


df_flow_temp


# GSHPs only

# In[113]:


df_gshps = df_all_resampled[df_all_resampled["HP_Type_2"] == "GSHP"]
df_gshps = df_gshps[df_gshps["External_Air_Temperature"] >= -7]
df_ext_temp, df_flow_temp = temp_vs_cop_calcs(df_gshps, show_plots=True, min_ext_air_temp = -10, max_ext_air_temp=18)


# In[114]:


df_ext_temp


# In[115]:


df_flow_temp


# ASHPs and GSHPs

# Noting that the GSHP sample size was much smaller than that of ASHPs, comparing the two 
# sets of results indicates that, as may be expected, the GSHP COP was much less temperature 
# dependent than that for ASHPs.
# 
# This is because the energy available in the GSHPs heat source (geothermal energy) does not 
# fluctuate as much as the external air as changes in weather are experienced. The COP still 
# decreased slightly, most likely as a result of the increasing flow temperature as the external 
# temperature reduced, however the impact was much lower.

# In[116]:


df_ashps_gshps = df_all_resampled[df_all_resampled["HP_Type_2"] != "Hybrid"]
df_all_resampled_ext_temp_stats, df_all_resampled_flow_temp_stats = temp_vs_cop_calcs(df_ashps_gshps, show_plots=True, min_ext_air_temp = -6, max_ext_air_temp=18, color_by="HP_Type_2")


# In[117]:


df_all_resampled_ext_temp_stats


# In[118]:


df_all_resampled_flow_temp_stats


# ASHPs split by refrigerant

# These figures show that all of the heat pumps operated with a similar efficiency between -7°C and 
# 3°C external temperature. Below -7°C, the sample sizes for all categories are small, hence the 
# data became spikier. Above 3°C, the performance of both R32 and R290 heat pumps continued 
# to rise at a steep gradient whereas R410a heat pumps performed with a lower efficiency. As 
# most of the UK heating season is above 3°C external temperature, this shows that the more 
# modern R32 and R290 heat pumps were better optimised for UK conditions however, that the 
# (very) cold weather COP across all of the ASHPs was consistent. 

# In[119]:


df_all_resampled_ext_temp_stats, df_all_resampled_flow_temp_stats = temp_vs_cop_calcs(df_ashps, show_plots=True, min_ext_air_temp = -10, max_ext_air_temp=18, color_by="HP_Refrigerant")


# In[120]:


df_all_resampled_ext_temp_stats


# In[121]:


df_all_resampled_flow_temp_stats


# In[122]:


# Set sample size so that scatter charts can generate without performance issues
if resolution == "30min":
    sample = 10000
else:
    sample = filtered_rows

df_sample = df_all_resampled.sample(sample)


# In[123]:


# Optional box plot chart to show external air temperature by year and month
px.box(
    df_sample.sort_values(["Timestamp", "Property_ID"], ascending=True),
    x="Month_name",
    y="External_Air_Temperature",
    width=800,
    title="Box plot of external air temperature by year and month",
    color="Year")


# In[124]:


# Optional chart to show average heat pump energy output by month of the year
px.bar(
    df_all_resampled.groupby(["Month_name", "Month"])["Heat_Pump_Energy_Output"].mean().reset_index().sort_values("Month"),
    x="Month_name",
    y="Heat_Pump_Energy_Output",
    title="Average Heat Pump Energy Output by Month",
    width=800)

