#!/usr/bin/env python
# coding: utf-8

# ### Comparing Designed and Actual Performance

# This section compares the SPF with the MCS listed Seasonal Coefficient of Performance 
# (SCOP) values. For this comparison the SPFH2 value is used as this represents the same 
# system boundary as the SCOP. 
# For a property to be included in this part of the analysis it was required to have a SPF which
# met the criteria to be included in the SPF analysis and have a SCOP listed 
# on the MCS installations database. Note that all listed SCOPs have been obtained from the 
# MCS installations database and are provided in the supplementary dataset.

# In[53]:


import pandas as pd
import os
import numpy as np
import analysis_functions as af
from tqdm import tqdm
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display
import chart_functions as cf

pio.renderers.default = "notebook"

# Set no limit to the number of columns displayed
pd.set_option("display.max_columns", None)

# Set plotly colour scheme
pio.templates["EoH"] = go.layout.Template(
    layout_colorway=["#%02x%02x%02x" % rgb_colour for rgb_colour in cf.colour_scheme().values()]
)
pio.templates.default = "EoH"

####-------------------
### DEFINITIONS: 

## Performance factor ranges for different time scales: 
# - For short time scales, higher variation in the performance factor are expected
spf_ranges = {
    "short": {"min": 0.75, "max": 7.5}, # short time periods: ~a day
    "medium": {"min": 0.9, "max": 6.5}, # medium tim periods: ~weeks or months
    "long": {"min": 1.5, "max": 5.0},   # long time periods: ~a year
}

## - Different dimensions for the figures based on number of plots showns (when facet argument used):
squared_dim_plots = {
    "1pic": {"h": 600, "w": 600},     # 1 squared plot
    "1pic_lg": {"h": 600, "w": 700},  # 1 squared plot with legend
    "2pic": {"h": 470, "w": 900},     # 2 squared plots
    "3pic": {"h": 450, "w": 1100},    # 3 squared plots
    "4pic": {"h": 400, "w": 1200},    # 4 squared plots
}

## - Readable labels to add to the plots:
lab_derived={"per_diff_scop": {"per_diff_scop": "Relative (%) difference (SPFH2-SCOP)/SCOP"},
             "spfh2_scop": {"spfh2_scop": "Difference (SPFH2-SCOP)"},
             "per_diff_flow_temp": {"per_diff_flow_temp": "Relative (%) difference (mean HPHFT-designed FT)/(designed FT)"}, 
             "wmhphft_ftemp": {"wmhphft_ftemp" : "Difference (mean HPHFT-designed FT)"}}

####-------------------
### PATHS:


# Specify files location paths:
location = os.environ["EoH"]
location_out = os.path.join(location, "processed")
location_out_cleaned = os.path.join(location_out, "cleaned")
# - And those of the main output clean data files:
location_out_cleaned_half_h = os.path.join(location_out_cleaned, "cleaned_half_hourly")

## Output path to store figures and tables/outputs used for inspection or to used in documentation:

location_outputs_analysis = os.path.join(location_out, "as_designed_SCOP_FT_comparison")
if not os.path.exists(location_outputs_analysis):
    os.mkdir(location_outputs_analysis)


# In[54]:


####-------------------
### LOADING DATA: 
# 
## a) Home sumamry stats with install information:
home_summary_ins = pd.read_csv(os.path.join(location_out, "home_summary_with_install_info.csv"))
#
## b) HP design data:
home_design = pd.read_excel(r"MCS ID and Design Details.xlsx", sheet_name = "data")

# - To know and compare dimensions of data:
print("The number of properties in the home summary file is "+str(home_summary_ins.shape[0])+
      ".\nThe number of properties present in the design details document is "+str(home_design.shape[0])+".\n")


# ##### Comparison of calculated SPFH2 within window with “as designed” SCOP

# In[55]:


####---------------
### EDA: Selecting the homes for which acceptable SPFH2 values could be calculated
#        And comparing the calculated values with the as designed SCOP

# - Merging both data frames: design information and reduced df with home summary info:
df_summ_desg_all = pd.merge(home_summary_ins, home_design, how="left", on="Property_ID")
#
# - Printing number of properties for which as designed SCOP is available:
scop_ava_tot = df_summ_desg_all[~df_summ_desg_all["SCOP"].isna()].shape[0]
per_scop = round(scop_ava_tot/len(df_summ_desg_all["Property_ID"].unique())*100, 1)
print("Number of properties for which as designed SCOP is available: "+str(scop_ava_tot)+" out of "+
      str(len(df_summ_desg_all["Property_ID"].unique()))+" ("+str(per_scop)+"%).\n")

# - Replace strings with NaN:
df_summ_desg_all = df_summ_desg_all.replace("not known|N/A|VALUE!", np.nan, regex=True)
# - Converting 'flow temperature' from int64 to float64
df_summ_desg_all["flow temperature"] = df_summ_desg_all["flow temperature"].astype('float')

# - Deriving number of properties included in the analysis:
tot_homes_analysed = df_summ_desg_all[df_summ_desg_all["Removed_from_analysis"] == False].shape[0]
# And %age of homes with SPFH2 in the acceptable window:
per_tot_homes = round(tot_homes_analysed/df_summ_desg_all.shape[0]*100, 1)

# - Keeping only the properties that were included in the analysis and 
# those with both calculated SPFH2 and as designed SCOP:
df_summ_desg = df_summ_desg_all[(df_summ_desg_all["Removed_from_analysis"] == False) & 
                                (~df_summ_desg_all["SPFH2"].isna()) & (~df_summ_desg_all["SCOP"].isna())].copy()
per_incl = round(len(df_summ_desg['Property_ID'].unique())/len(df_summ_desg_all["Property_ID"].unique())*100, 1)

####---------------
## Printing some results, to convert into Markdown if needed:
#
# - Checking that number of properties after merging data frame equals that of the summary with info data frame:
if (len(df_summ_desg_all["Property_ID"]) != len(df_summ_desg_all["Property_ID"])):
    print(f"The merged data frame does not contain the same number of properties as the summary file.")
else:
    print(f"Merge of data from home summary and designed information was done correctly.\n")
    print("A total of "+str(tot_homes_analysed)+" properties were included in the EoH analysis, ~"+
          str(per_tot_homes)+"% of the original list.")
    print(f"From these, {len(df_summ_desg['Property_ID'])} properties ("+
          str(per_incl)+
          "% of total) had an available 'as designed' SCOP value and therefore were included in this analysis.")


# For 92% of the properties considered, the SPFH2 was lower than the SCOP. The mean 
# difference between the two was -0.66, meaning the SPFH2 was 18% lower than the SCOP.
# 
# Reviewing the results, whilst this study has found that for many homes the real-world 
# performance of heat pumps is good, it is clear that there is a systematic overestimation of heat 
# pump efficiencies at the design stage.

# In[56]:


####---------------
### EDA: 
#
# - Comparing SCOP with SPFH2:
#
# a) Difference
df_summ_desg["spfh2_scop"] = df_summ_desg["SPFH2"] - df_summ_desg["SCOP"]
#
# b) Percentage difference with respect the designed SCOP
df_summ_desg["per_diff_scop"] = (df_summ_desg["spfh2_scop"]) / df_summ_desg["SCOP"]*100
#
# c) Ratio
df_summ_desg["sphh2/scop"] = df_summ_desg["SPFH2"] / df_summ_desg["SCOP"]
df_summ_desg["scop/sphh2"] = df_summ_desg["SCOP"]/df_summ_desg["SPFH2"]

##-----
# - Comparing manufactured 'flow temperature' with 'window_mean_Heat_Pump_Heating_Flow_Temperature':
#
# a) Difference
df_summ_desg["wmhphft_ftemp"] = (df_summ_desg["window_mean_Heat_Pump_Heating_Flow_Temperature"] - df_summ_desg["flow temperature"])
#
# b) Percentage difference with respect the designed 'flow temperature'
df_summ_desg["per_diff_flow_temp"] = (df_summ_desg["wmhphft_ftemp"]) / df_summ_desg["flow temperature"]*100
#
# c) Ratio
df_summ_desg["wmhphft/ftemp"] = (df_summ_desg["window_mean_Heat_Pump_Heating_Flow_Temperature"] / df_summ_desg["flow temperature"])

# - Creating a dummy variable for grouping purposes
df_summ_desg["all"] = "EoH"

##--------
### Deriving descriptive stats for SPFH2 vs. SCOP:
#
# - Considering the whole sample:
df_scop_spfh2_stats = df_summ_desg.groupby("all")[['SCOP', 'SPFH2']].describe()#.T
#
# - By HP type:
df_scop_spfh2_hp_stats = df_summ_desg.groupby('HP_Type')[['SCOP', 'SPFH2']].describe()

##------
### Deriving descriptive stats for the relative difference and differce:
print(f"Relative difference and difference of calculated SPFH2 and SCOP for whole sample:")
# - Considering the whole sample:
df_scop_spfh2_diff = df_summ_desg.groupby("all")[['per_diff_scop', 'spfh2_scop']].describe()
##-- Table
display(df_scop_spfh2_diff.rename(columns={'per_diff_scop':'(SPFH2-SCOP)/SCOP', 'spfh2_scop':'SPFH2-SCOP'}))

# --- Plotting the results
# - BOX PLOT of the percentage difference between calculated SPFH2 and as designed SCOP: 
fig_bp_perd =px.box(df_summ_desg, y="per_diff_scop", points="all", 
            hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'], 
            height=500, width=550, labels=lab_derived["per_diff_scop"])
fig_bp_perd.update_traces(quartilemethod="inclusive")
fig_bp_perd.show()
path_out=os.path.join(location_outputs_analysis, "Percentage_diff_SPFH2_SCOP.png")
pio.write_image(fig_bp_perd, path_out, format="png", engine="orca")

##--
# - BOX PLOT of the difference between calculated SPFH2 and designed SCOP:
fig_bp_diff =px.box(df_summ_desg, y="spfh2_scop", points="all", 
            hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'], 
            height=500, width=550, labels=lab_derived["spfh2_scop"])
fig_bp_diff.update_traces(quartilemethod="inclusive")
fig_bp_diff.show()
path_out=os.path.join(location_outputs_analysis, "Difference_SPFH2_SCOP.png")
pio.write_image(fig_bp_diff, path_out, format="png", engine="orca")

##---------
# - By HP type 2:
df_scop_spfh2_hp_diff = df_summ_desg.groupby('HP_Type_2')[['per_diff_scop', 'spfh2_scop']].describe()
#
print(f"Relative difference and difference of calculated SPFH2 and SCOP by Heat Pump Type:")
##-- Table
display(df_scop_spfh2_hp_diff.rename(columns={'per_diff_scop':'(SPFH2-SCOP)/SCOP', 'spfh2_scop':'SPFH2-SCOP'}))

##--
# - BOX PLOT of the percentage difference between calculated SPFH2 and designed SCOP by HP type:
fig_bp_perdiff =px.box(df_summ_desg[df_summ_desg["HP_Type_2"]!="GSHP"], x="HP_Type_2", y="per_diff_scop", color="HP_Type_2", points="all", 
            hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'],
            labels=lab_derived["per_diff_scop"])#, facet_row='HP_Refrigerant')
fig_bp_perdiff.update_xaxes(title_text="Heat Pump Type")
fig_bp_perdiff.update_layout(showlegend=False)
fig_bp_perdiff.update_traces(quartilemethod="inclusive")
fig_bp_perdiff.show()
path_out=os.path.join(location_outputs_analysis, "Percentage_diff_SPFH2_SCOP_by_HP_Type.png")
pio.write_image(fig_bp_perdiff, path_out, format="png", engine="orca")

##--
# - BOX PLOT of the difference between calculated SPFH2 and designed SCOP by HP type:
fig_bp_absdiff =px.box(df_summ_desg[df_summ_desg["HP_Type_2"]!="GSHP"], x="HP_Type_2", y="spfh2_scop", color="HP_Type_2", points="all", 
            hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'],
            labels=lab_derived["spfh2_scop"])
fig_bp_absdiff.update_xaxes(title_text="Heat Pump Type")
fig_bp_absdiff.update_layout(showlegend=False)
fig_bp_absdiff.update_traces(quartilemethod="inclusive")
fig_bp_absdiff.show()
path_out=os.path.join(location_outputs_analysis, "Difference_SPFH2_SCOP_by_HP_Type.png")
pio.write_image(fig_bp_absdiff, path_out, format="png", engine="orca")


# In[57]:


## Determining the percentage of properties for which the difference metric was negative by:
# a) HP type & b) HP refrigerant:
metrics = ["all", "HP_Type_2", "HP_Refrigerant"]

print("Percentage of properties for which SPFH2 < as designed SCOP considering the following samples:")
for col in metrics:
    for hp_tp in df_summ_desg[col].unique():
        fg=df_summ_desg[(df_summ_desg["spfh2_scop"]<0) & (df_summ_desg[col] == hp_tp)]
        fg2=df_summ_desg[df_summ_desg[col] == hp_tp]
        print(hp_tp)
        print(str(round(fg.shape[0]/fg2.shape[0]*100,1))+"%")


# In[58]:


###---------------------------
## Printing results into csv file to easy copy them into the report:
df_scop_spfh2_sum = df_summ_desg.groupby("all")[['SPFH2', 'SCOP']].describe()
df_scop_spfh2_sum.T.to_csv(os.path.join(location_outputs_analysis, "scop_spfh2_all_sample.csv"), float_format='%.{}f'.format(2))
#
# a) Considering all the sample:
df_scop_spfh2_diff.rename(columns={'per_diff_scop':'(SPFH2-SCOP)/SCOP', 'spfh2_scop':'SPFH2-SCOP'})
df_scop_spfh2_diff.T.to_csv(os.path.join(location_outputs_analysis, "comparison_scop_spfh2_all_sample.csv"), float_format='%.{}f'.format(2))
#
# b) By HP type:
df_scop_spfh2_hp_diff.rename(columns={'per_diff_scop':'(SPFH2-SCOP)/SCOP', 'spfh2_scop':'SPFH2-SCOP'})
df_scop_spfh2_hp_diff.T.to_csv(os.path.join(location_outputs_analysis, "comparison_scop_spfh2_byHP.csv"), float_format='%.{}f'.format(2))
#
# c) By HP Refrigerant:
df_scop_spfh2_ref_diff = df_summ_desg.groupby('HP_Refrigerant')[['per_diff_scop', 'spfh2_scop']].describe()
df_scop_spfh2_ref_diff.T.to_csv(os.path.join(location_outputs_analysis, "comparison_scop_spfh2_byRefrigerant.csv"), float_format='%.{}f'.format(2))

###---------------------------
## And priting the some additional stats to populate the EoH report tables:
#
# - Defining the confidence level:
ci_used = 0.95
# - Defining the metrics to report:
stats_metric = ['SPFH2', 'SCOP', 'spfh2_scop', 'per_diff_scop']

# a) Considering all the sample:
print("Entire sample: ")
for metric_val in stats_metric:
    sample_mean, ci_lower, ci_upper, sample_median, iqr = af.iqr_confidence_interval(df_summ_desg, metric=metric_val, confidence_level=ci_used, digits=2)

# b) By HP type:
for id in df_summ_desg["HP_Type_2"].unique():
    df_filtered = df_summ_desg[df_summ_desg["HP_Type_2"] == id]
    print("By HP Type: "+ id)
    for metric_val in stats_metric[2:5]:
        sample_mean, ci_lower, ci_upper, sample_median, iqr = af.iqr_confidence_interval(df_filtered, metric=metric_val, confidence_level=ci_used, digits=2)

# c) By HP Refrigerant:
for id in df_summ_desg["HP_Refrigerant"].unique():
    df_filtered = df_summ_desg[df_summ_desg["HP_Refrigerant"] == id]
    print("By HP Refrigerant: "+ id)
    for metric_val in stats_metric[2:5]:
        sample_mean, ci_lower, ci_upper, sample_median, iqr = af.iqr_confidence_interval(df_filtered, metric=metric_val, confidence_level=ci_used, digits=2)


# As the spread of results was quite wide across all heat pump types, any correlation between the 
# SPFH2 and SCOP was explored to understand whether the SCOP was routinely higher by a 
# similar amount across the sample or whether the differences were more random. The scatter 
# plot shown below plots each SPFH2 result against the properties designed SCOP. To 
# assess if there was any relation between the two variables compared, an Ordinary Least 
# Square (OLS) regression model was implemented, this is shown in grey. In red, an y=x line is 
# shown to help visualise and interpret the results.
# 
# As indicated before, most of the data points fall below the y=x red line, i.e., SPFH2 was lower 
# than ‘as designed’ SCOP. The R2 of the OLS regression model is 0.05 (p-value = 0.00), which 
# confirms that there was little to no correlation between the ‘as designed’ SCOP and derived 
# SPFH2.
# There are two conclusions which can be drawn from this. Firstly, the fact 'as designed' SCOPs 
# consistently overestimate performance indicates there is a systematic optimism bias in 'as designed' SCOP calculations that needs addressing so consumers are given a realistic picture 
# of the performance they can expect. 
# 
# Just as importantly, the lack of correlation suggests that the current methods for estimating 'as 
# designed' SCOP not only overestimate performance, but also fail to predict whether this heat 
# pump installation will perform well compared to others. To illustrate, this data suggests that we 
# can't assume that a system with an 'as designed' SCOP of 4 will perform any better than one 
# with an 'as designed' SCOP of 3. This shows the current method for selecting SCOP at the 
# design stage is likely to provide an unrealistic estimate which is of questionable use to 
# consumers.

# In[59]:


####---------------
### PRINTING:

# - Creating a CSV file containing the fields calculated (renaming and removing columns not relevant)
# 
# Derived/calculated columns to keep:
cols_2keep_cal = ['Property_ID', 'spfh2_scop', 'per_diff_scop', 'scop/sphh2']
#
# - Selecting only the SCOP results from the comparison:
df_comp_results = df_summ_desg.loc[:, cols_2keep_cal].rename(
    columns={'spfh2_scop':'SPFH2-SCOP', 'per_diff_scop':'(SPFH2-SCOP)/SCOP', 'scop/sphh2':'SCOP/SPFH2'})
df_summ_desg_all = df_summ_desg_all.rename(columns={'SCOP':'As designed SCOP', 'flow temperature':'As designed flow temperature'})
df_summ_desg_prt = pd.merge(df_summ_desg_all, df_comp_results, how="left", on="Property_ID")
df_summ_desg_prt.to_csv(os.path.join(location_out, "home_summary_with_as_design_info.csv"), index=False)

####---------------
### PLOTTING: 

## - SCATTER PLOT with trendline: SPFH2 = f(SCOP)
fig_sc_spfh2_scop =px.scatter(df_summ_desg, x="SCOP", y="SPFH2", #color="#636363",
                 hover_data=['Property_ID', 'HP_Type','window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'], 
                 height=squared_dim_plots["1pic"]["h"], width=squared_dim_plots["1pic"]["w"], 
                 trendline='ols', trendline_color_override="#7f7f7f")#, facet_row='HP_Type')
# - DEFINING and adding a y=x line based on spf ranges:
line_trace = go.Scatter(x=[spf_ranges["long"]["min"], spf_ranges["long"]["max"]], y=[spf_ranges["long"]["min"], spf_ranges["long"]["max"]],
                        mode='lines', name="y=x", line=dict(color='red', width=0.7))
line_trace.update(legendgroup="y=x", showlegend=False)
fig_sc_spfh2_scop.add_trace(line_trace)
#
# - Adding labels to the two lines
# a) y=x label:
fig_sc_spfh2_scop.add_trace(go.Scatter(
    x=[2.3], y=[2.5],
    mode="text", name="y=x",
    text=["y=x"], textposition="top center",
    textfont=dict(
        size=14, color="red"
    )
))
# b) OSL label:
fig_sc_spfh2_scop.add_trace(go.Scatter(
    x=[4.4, 4.4], y=[3.65, 3.48],
    mode="text", name="OLS",
    text=["OLS", "trendline"],
    textposition="bottom right",
    textfont=dict(
        size=14.5, color="#7f7f7f"
    )
))
# - Removing the legend:
fig_sc_spfh2_scop.update_layout(showlegend=False)
fig_sc_spfh2_scop.show()
path_out=os.path.join(location_outputs_analysis, "SPFH2_vs_SCOP_withOLS.png")
pio.write_image(fig_sc_spfh2_scop, path_out, format="png", engine="orca")

model = px.get_trendline_results(fig_sc_spfh2_scop)
results = model.iloc[0]["px_fit_results"]
print(f"r^2: {round(results.rsquared,2)}")
print(f"p-value: {round(results.pvalues[1], 2)}")


# In[60]:


####---------------
### OTHER PLOTS: - to remove if not usefull

##----
# - SCATTER PLOT with trendline by HP type: SPFH2 = f(SCOP)
fig_sc_hp_spfh2_scop =px.scatter(df_summ_desg, y="SPFH2", x="SCOP",                                 
                 hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'], 
                 height=squared_dim_plots["4pic"]["h"], width=squared_dim_plots["4pic"]["w"], 
                 trendline='ols', trendline_color_override="#7f7f7f", facet_col='HP_Type_2')
# - Adding a y=x line based on spf ranges:
line_trace = go.Scatter(x=[spf_ranges["long"]["min"], spf_ranges["long"]["max"]], y=[spf_ranges["long"]["min"], spf_ranges["long"]["max"]],
                        mode='lines', name="y=x", line=dict(color='red', width=0.7))
line_trace.update(legendgroup="y=x", showlegend=False)
# - Add the line trace to all figure:
fig_sc_hp_spfh2_scop.add_trace(line_trace, row="all", col="all", exclude_empty_subplots=True)
fig_sc_hp_spfh2_scop.show()
path_out=os.path.join(location_outputs_analysis, "SPFH2_vs_SCOP_withOLS_by_HP_Type.png")
pio.write_image(fig_sc_hp_spfh2_scop, path_out, format="png", engine="orca")

##----
# - SCATTER PLOT with trendline by HP Refrigerant: SPFH2 = f(SCOP)
fig_sc3_ref =px.scatter(df_summ_desg, y="SPFH2", x="SCOP", 
                 hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'], 
                 height=squared_dim_plots["4pic"]["h"], width=squared_dim_plots["4pic"]["w"], 
                 trendline='ols', trendline_color_override="#7f7f7f", facet_col='HP_Refrigerant')
line_trace.update(legendgroup="y=x", showlegend=False)
fig_sc3_ref.add_trace(line_trace, row="all", col="all", exclude_empty_subplots=True)

fig_sc3_ref.show()
path_out=os.path.join(location_outputs_analysis, "SPFH2_vs_SCOP_withOLS_by_HP_Refrigerant.png")
pio.write_image(fig_sc3_ref, path_out, format="png", engine="orca")


##----
# - BOX PLOTS with percentage diff by HP Refrigerant
fig_bp_ref =px.box(df_summ_desg, x="HP_Refrigerant", y="per_diff_scop", color="HP_Refrigerant", points="all", 
            hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'],
            labels=lab_derived["per_diff_scop"])
fig_bp_ref.update_xaxes(title_text="Heat Pump Refrigerant")
fig_bp_ref.update_layout(showlegend=False)
fig_bp_ref.update_traces(quartilemethod="inclusive")
fig_bp_ref.show()
path_out=os.path.join(location_outputs_analysis, "Percentage_diff_SPFH2_SCOP_by_HP_Refrigerant.png")
pio.write_image(fig_bp_ref, path_out, format="png", engine="orca")

##----
# - BOX PLOTS with difference by HP Refrigerant
fig_bp_ref =px.box(df_summ_desg, x="HP_Refrigerant", y="spfh2_scop", color="HP_Refrigerant", points="all", 
            hover_data=['Property_ID', 'window_mean_Heat_Pump_Heating_Flow_Temperature', 'flow temperature'],
            labels=lab_derived["spfh2_scop"])
fig_bp_ref.update_xaxes(title_text="Heat Pump Refrigerant")
fig_bp_ref.update_layout(showlegend=False)
fig_bp_ref.update_traces(quartilemethod="inclusive")
fig_bp_ref.show()
path_out=os.path.join(location_outputs_analysis, "Difference_SPFH2_SCOP_by_HP_Refrigerant.png")
pio.write_image(fig_bp_ref, path_out, format="png", engine="orca")



# ##### Comparison of heat pump heating flow temperature (HPHFT) within window with “as designed” Flow Temperature (FT)

# In[61]:


####---------------
### EDA: Selecting the properties included in the analysis (and with window mean flow temperature) 
#        And comparing the HPH flow temperature values with the as designed flow temperature.

# - Calculating and printing number of properties for which as designed flow temperature is available:
ft_ava_tot = df_summ_desg_all[~df_summ_desg_all["As designed flow temperature"].isna()].shape[0]
per_ft = round(ft_ava_tot/len(df_summ_desg_all["Property_ID"].unique())*100, 1)
print("Number of properties for which as designed flow temperature (FT) is available: "+str(ft_ava_tot)+" out of "+
      str(len(df_summ_desg_all["Property_ID"].unique()))+" ("+str(per_ft)+"%).\n")

# - Keeping only the properties that were included in the analysis and 
# those with both calculated window mean HPH FT and as designed FT:
df_summ_desg_ft = df_summ_desg_all[(home_summary_ins["Removed_from_analysis"] == False) & 
                                (~df_summ_desg_all["window_mean_Heat_Pump_Heating_Flow_Temperature"].isna()) & (~df_summ_desg_all["As designed flow temperature"].isna())].copy()
per_incl_fl = round(len(df_summ_desg_ft['Property_ID'].unique())/len(df_summ_desg_all["Property_ID"].unique())*100, 1)

####---------------
## Printing some results, to convert into Markdown if needed:
#
# - Checking that number of properties after merging data frame equals that of the summary with info data frame:
print("From the "+str(tot_homes_analysed)+f" properties included in the analysis, {len(df_summ_desg_ft['Property_ID'])}"
      + " ("+str(per_incl_fl)+"% of total) had an available 'as designed' FT value.")


# We explored which quantile of the measured HPH flow temperature gets closer to the 'as designed' flow temperature for each property. For this analysis, we used their 30 minutes clean data, filtering out the HPH flow temperature observations outside the selected window as well as any null values. 
# 
# The HPHFT quantile values ranging from 5% to 100% (where 100% is the maximum registered) were calculated for each property. Then, all the flow temperature values for each quantile and property were compared to their 'as designed' FT by deriving the difference of these metrics. The quantile value which HPHFT was closest to the 'as designed' FT was selected, i.e., the one for which the deviation or difference with respect the 'as designed' FT was lowest (minimum). 

# In[62]:


####-------------------
## LOADING DATA: Reading the cleaned 30-min clean data for only the homes included in the analysis and with 'as designed' FT:

# - Initialising the data frame containing that data:
df_30m_clean = pd.DataFrame()
# - Specifying the data format:
file_format = "parquet"
# - Listing the relevant of columns from the HP summary data frame to keep for this analysis:
designed_field_kept = ["Property_ID", "HP_Type", "HP_Type_2", "window_mean_Heat_Pump_Heating_Flow_Temperature", "window_start", "window_end", "As designed flow temperature"]

## Loop across the 30min clean/processed data:
for home_id in tqdm(sorted("Property_ID="+df_summ_desg_ft["Property_ID"].unique())):
    
    #  - Name of the file: concatenating home ID with file format
    file_home = home_id + "." + file_format
    #
    # - Loading the 30 min data:
    tem_df = pd.read_parquet(os.path.join(location_out_cleaned_half_h, file_home))
    
    # - Adding the property ID as a column:
    tem_df["Property_ID"] = home_id[-7:]
    
    # - Inner joining the 30 min data frame with the home installation/designed detailed summary:
    temp_merg = pd.merge(tem_df, df_summ_desg_ft[designed_field_kept], on="Property_ID")
   
    # - Filtering out the data outside the analysis window selected:
    temp_merg_f = temp_merg[(temp_merg['Timestamp'].dt.strftime('%Y-%m-%d') >= temp_merg["window_start"]) & 
                             (temp_merg['Timestamp'].dt.strftime('%Y-%m-%d') <= temp_merg["window_end"])]

    # Appending the dataframes: 
    df_30m_clean = pd.concat([df_30m_clean, temp_merg_f]) 
    


# In[63]:


####-------------------
### EDA: Filtering out NaN data and deriving the quantile values for the HPH FT for each property:

## Keeping only the observations for which heat pump heating flow temperature (HPHFT) is not NaN:
wind_temp_nona = df_30m_clean[~(df_30m_clean["Heat_Pump_Heating_Flow_Temperature"].isna())].copy()
                          
## Calculating a range of quantiles, from 0.05 to 1, for each property:
# - Parameters for the quantile vector range and vector:
end_q = [0.85,1.0]
step_q = [0.05,0.01]
q_range_coved_lg = np.arange(0.05, end_q[0]+step_q[0], step_q[0])
q_range_coved_gn = np.arange(0.90, end_q[1]+step_q[1], step_q[1])
q_range_coved = sorted(np.concatenate([np.around(q_range_coved_lg, 2), np.around(q_range_coved_gn, 2)]))

_## Calculating the quantiles for each building
temp_rg_quantiles = wind_temp_nona.groupby(designed_field_kept)['Heat_Pump_Heating_Flow_Temperature'].quantile(q_range_coved)
temp_rg_quantiles = temp_rg_quantiles.reset_index()
#
# - Calculating the absolute difference between the quantile values of the HPHFT and the as designed FT:
temp_rg_quantiles["Absolute difference HPHFT to as designed FT"] = np.abs(temp_rg_quantiles["Heat_Pump_Heating_Flow_Temperature"]-temp_rg_quantiles["As designed flow temperature"])
#
# - Renaming the quantile column:
col_name_hphft = 'HPHFT_quantile'  
temp_rg_quantiles.columns = [col_name_hphft if col.startswith('level_') else col for col in temp_rg_quantiles.columns]

## Identifying the minimum value for the absolute difference between HPH FT and as designed FT for each property and renaming that column:
df_fl_gr = temp_rg_quantiles.groupby("Property_ID").agg(
    min_dif_hph_ft=('Absolute difference HPHFT to as designed FT', 'min'),
).reset_index().rename(columns={'min_dif_hph_ft':'Absolute difference HPHFT to as designed FT'})


## Joining with the data frame containing the quantile values to keep a single entry per property with the quantile closest to the as designed FT:
df_fl_asdesig = pd.merge(temp_rg_quantiles, df_fl_gr, on=["Property_ID", "Absolute difference HPHFT to as designed FT"])
if (df_fl_asdesig.shape[0] > len(temp_rg_quantiles["Property_ID"].unique())):
    # - Counting number of observations per Property ID
    num_entries = df_fl_asdesig['Property_ID'].value_counts().reset_index()
    # - Renaming columns:
    num_entries.columns = ['Property_ID', 'num_obs']
    # - Keeping only the Property IDs of the houses with 'num_obs' > 1:
    multiple_entries = num_entries[num_entries["num_obs"]>1]
    display(multiple_entries)
    list_of_prop = pd.Series(multiple_entries["Property_ID"].unique())
    print("Error: More than two quantile values assigned to the following properties: "+
          list_of_prop.str.cat(sep=', ', na_rep=''))
    print("Properties removed from the analysed sample.")
    #
    # - Removing those properties with multiple qunatile entries and not using them for the analysis:
    df_fl_asdesig_f = df_fl_asdesig[~(df_fl_asdesig["Property_ID"].isin(list_of_prop))].copy()
    temp_rg_quantiles_f = temp_rg_quantiles[~(temp_rg_quantiles["Property_ID"].isin(list_of_prop))].copy()
    print("Final sample size: "+str(len(df_fl_asdesig_f["Property_ID"].unique())))
else:
    df_fl_asdesig_f = df_fl_asdesig.copy()
    temp_rg_quantiles_f = temp_rg_quantiles.copy()
                                          

## Writing output for inspection:
df_fl_asdesig_f.to_csv(os.path.join(location_outputs_analysis, "EoH-HPH_vs_as-designed_FT_comparison.csv"), index=False)
# - And with all quantile values for all properties analysed:
temp_rg_quantiles.to_csv(os.path.join(location_outputs_analysis, "EoH-HPH_vs_as-designed_FT_comparison_all-quantiles.csv"), index=False)


# To assess the whole dataset, the measured heat pump heating flow temperature percentiles were 
# calculated from 1% to 100% for each property (where 100% was the maximum recorded 
# temperature). Then, all the flow temperature values for each quantile and property were compared 
# to their 'as designed' flow temperature by deriving the difference of these metrics. The measured 
# flow temperature quantile value closest to the 'as designed' flow temperature (i.e. the one for 
# which the difference was minimum showing the lowest deviation with respect the 'as designed' 
# flow temperature) was identified and selected. The histogram below summarises the 
# results of this calculation.
# 
# For 119 properties (25% of total) the measured flow temperature quantile value closest to the 
# 'as designed' flow temperature was 99%.
# 
# As well as evaluating the quantile where the designed flow temperature lies, it was necessary to 
# assess any correlation between designed flow temperature and actual flow temperature. As the 
# designed flow temperature most commonly fell within the 99
# th percentile, each heat pumps 
# designed flow temperature was plotted against the 99
# th percentile value of actual flow 
# temperature. The resulting scatter graph is shown below.
# 
# An OLS regression model was applied to explore any potential correlation between these metrics. 
# However, the R2 confirmed that 'as designed' flow temperature and the 99th percentile measured 
# flow temperature values are not related.
# 
# This result again indicates one of two things. Either that design methods and resulting designed 
# flow temperatures failed to predict the actual operation of the heat pump or (perhaps more 
# likely) that heat pump system operation has not aligned closely enough with the system design.
# This may be down to heat pumps being commissioned to operate at temperatures different to 
# those in the design or due to design error resulting in adjustment of the settings part way 
# through operation. 
# 
# One cause of design error could be the overestimation of building heat loss and 
# underestimation of thermal gains, this would result in designed flow temperatures therefore 
# being higher than necessary. Equally, using a heat pump more sporadically than was assumed 
# or intended in the design may result in higher operating flow temperatures (due to load 
# compensation control) and therefore the designed flow temperature being lower than the actual.
# 
# Ultimately, whilst important for particularly cold days, the 99th percentile 'design flow 
# temperature' is less important to the real-world efficiency of the heat pump than the 'typical' (e.g. 
# mean/median) flow temperature and therefore when considering how to optimally design a 
# heating system there should be increased focus on designing (and documenting) weather 
# compensation curves rather than just the 99th percentile flow temperature. This would hopefully 
# bring the designed performance estimates closer to those experienced in-situ, offering 
# consumers greater clarity on their expected energy use.

# In[64]:


####-------------------
### REPORTING: Histograms & printing information

###-------------
## Calculating some summary statistics of the data grouped by HPHFT quantile:
sum_fl_quantile = df_fl_asdesig_f.groupby(["HPHFT_quantile"]).agg(
    num_properties=('Property_ID','count'),
    window_mean_HPHFT=('window_mean_Heat_Pump_Heating_Flow_Temperature', 'mean'),
    mean_as_designed_FT=('As designed flow temperature','mean'),
    mean_hphft = ('Heat_Pump_Heating_Flow_Temperature', 'mean'),
    mean_abs_diff_hphft_desft=('Absolute difference HPHFT to as designed FT', 'mean')).reset_index()
sum_fl_quantile["per_total_properties"] = round(sum_fl_quantile["num_properties"]/sum_fl_quantile['num_properties'].sum()*100,2)

# - Most common quantile closest to the 'as designed' flow temperature:
max_prop_num_qt = sum_fl_quantile["num_properties"].max()
max_pert_tot_qt = sum_fl_quantile.loc[(sum_fl_quantile["num_properties"] == max_prop_num_qt),"per_total_properties"].mean()
closest_q_ft = sum_fl_quantile.loc[(sum_fl_quantile["num_properties"] == max_prop_num_qt),"HPHFT_quantile"].unique()
#
print("For "+str(max_prop_num_qt)+" properties ("+str(max_pert_tot_qt)+
      "% of total) the HPHFT quantile value closest to the 'as designed' FT is: "+str(closest_q_ft)+".\n")


###-------------
## VISUALISATION:

## Creating a histogram with the distribution of the closest HPHFT quantile to the 'as designed' FT.
# - Customised bin ranges and values:
bin_edges = q_range_coved_gn
# - Defining a function to convert numeric quantile values into discrete (chr) values, grouping those <=90%:
def custom_binning(x):
    if x in np.round(np.arange(0.05, 0.91, 0.05),2):
        return ' <=0.90'  
    else:
        return str(x)

binned_data_df = [custom_binning(x) for x in df_fl_asdesig_f["HPHFT_quantile"]]
fig_q_hist = px.histogram(binned_data_df, x=sorted(binned_data_df),
                   title="HPH flow temperature percentile closest to 'as designed' FT",
                   labels={'binned_data':"HPH flow temperature percentile closest to 'as designed' flow temperature"}, 
                   opacity=0.75,text_auto=True, 
                   #category_orders={'binned_data_df': ['<=90','91','92','93','94','95','96','97','98','99','100']}
                   )
fig_q_hist.update_layout(xaxis=dict(dtick=0.01))
fig_q_hist.update_xaxes(title_text="Percentile")
fig_q_hist.update_traces(showlegend=False)
fig_q_hist.show()
path_out=os.path.join(location_outputs_analysis, "Histogram_closest_HPHFT_quantile_to_FT.png")
pio.write_image(fig_q_hist, path_out, format="png", engine="orca")

# - Per HP type:
fig_q_hp_hist = px.histogram(df_fl_asdesig_f["HPHFT_quantile"], color=df_fl_asdesig_f["HP_Type_2"], nbins=30, 
                   title="HPH flow temperature percentile closest to 'as designed' FT",
                   labels={'value':"HPH flow temperature percentile closest to 'as designed' flow temperature"}, 
                   opacity=0.75,text_auto=True
                   )
fig_q_hp_hist.show()
#
path_out=os.path.join(location_outputs_analysis, "Histogram_closest_HPHFT_quantile_to_FT_by_HP_Type.png")
pio.write_image(fig_q_hp_hist, path_out, format="png", engine="orca")

##-------------
# - Scatter plots showing the HPHFT for a specific quantile (INDICATED BELOW) vs the 'as designed' FT:
quant_selec = 0.99  #quantile selected
# - Tittle of the plot:
titl_quant= "HPHFT "+str(int(quant_selec*100))+"th percentile value vs. the 'as designed' FT"
# - Selecting the HPHFT value for the specific quantile
temp_rg_xquantile = temp_rg_quantiles_f[(round(temp_rg_quantiles_f["HPHFT_quantile"],2) == quant_selec)]
# 
print("Checking: List of properties: "+str(len(temp_rg_xquantile["Property_ID"].unique())))
#
fig_sc_ft_xql =px.scatter(temp_rg_xquantile, x="As designed flow temperature", y="Heat_Pump_Heating_Flow_Temperature",
                 hover_data=['Property_ID', 'HP_Type','window_mean_Heat_Pump_Heating_Flow_Temperature'], 
                 height=squared_dim_plots["1pic"]["h"], width=squared_dim_plots["1pic"]["w"],
                 trendline='ols', trendline_color_override="#7f7f7f", title=titl_quant)
# - DEFINING and adding a y=x line:
line_trace = go.Scatter(x=[32, 70], y=[32, 70], mode='lines', name="y=x", line=dict(color='red', width=0.7))
line_trace.update(legendgroup="y=x", showlegend=False)
fig_sc_ft_xql.add_trace(line_trace)
# - Adding labels:
# a) y=x label:
fig_sc_ft_xql.add_trace(go.Scatter(
    x=[38], y=[35],
    mode="text", name="y=x",
    text=["y=x"], textposition="top center",
    textfont=dict(size=14, color="red")
))
# b) OSL label:
fig_sc_ft_xql.add_trace(go.Scatter(
    x=[60, 60], y=[49, 47],
    mode="text", name="OLS",
    text=["OLS", "trendline"],
    textposition="bottom right",
    textfont=dict(
        size=14.5, color="#7f7f7f"
    )
))
# - Axis lables:
fig_sc_ft_xql.update_yaxes(title_text="HPHFT "+str(int(quant_selec*100))+"th percentile")
fig_sc_ft_xql.update_xaxes(title_text="'as designed' flow temperature °C")
# - Removing the legend:
fig_sc_ft_xql.update_layout(showlegend=False)
fig_sc_ft_xql.show()
#
file_out_xquant = "HPHFT_"+str(int(quant_selec*100))+"quantile_vs_FT_withOLS.png"
path_out=os.path.join(location_outputs_analysis, file_out_xquant)
pio.write_image(fig_sc_ft_xql, path_out, format="png", engine="orca")


##-------------
# - Boxplot with the selected quantile HPHFT value by pre-defined 'as designed' FT group:
#
temp_rg_xquantile.loc[:,"designed_FT_gr"] = np.where(temp_rg_xquantile["As designed flow temperature"] <= 45, "<=45",
                                               np.where((temp_rg_xquantile["As designed flow temperature"] > 45) & 
                                                        (temp_rg_xquantile["As designed flow temperature"] <= 50), "50", "55>="))
#
fig_qn_bx =px.box(temp_rg_xquantile, x="designed_FT_gr", y="Heat_Pump_Heating_Flow_Temperature", points="all",
            hover_data=['Property_ID', 'Heat_Pump_Heating_Flow_Temperature', 'As designed flow temperature'],
            category_orders={'designed_FT_gr': ['<=45', '50', '55>=']})
fig_qn_bx.update_yaxes(title_text="HPHFT "+str(int(quant_selec*100))+"th percentile")
fig_qn_bx.update_xaxes(title_text="'as designed' flow temperature groups °C")
fig_qn_bx.update_layout(showlegend=False)
fig_qn_bx.update_traces(quartilemethod="inclusive")
fig_qn_bx.show()
file_out_bxquant = "Boxplot_HPHFT_"+str(int(quant_selec*100))+"quantile_vs_FT.png"
path_out=os.path.join(location_outputs_analysis, file_out_bxquant)
pio.write_image(fig_qn_bx, path_out, format="png", engine="orca")


##-------------
# - Scatter plot showing the HPHFT quantile value closest to the 'as designed' FT:
fig_sc_ft_ql =px.scatter(df_fl_asdesig_f, x="As designed flow temperature", y="HPHFT_quantile", color="HP_Type_2",
                 hover_data=['Property_ID', 'HP_Type','window_mean_Heat_Pump_Heating_Flow_Temperature'], 
                 height=squared_dim_plots["1pic"]["h"], width=squared_dim_plots["1pic"]["w"],
                 title="Quantile of the HPHFT value closest to the 'as designed' FT")
# - Axis lables
fig_sc_ft_ql.update_yaxes(title_text="Quantile value")
fig_sc_ft_ql.update_xaxes(title_text="'as designed' flow temperature °C")
fig_sc_ft_ql.show()
path_out=os.path.join(location_outputs_analysis, "HPHFT_quantile_vs_FT_by_HP_Type.png")
pio.write_image(fig_sc_ft_ql, path_out, format="png", engine="orca")

