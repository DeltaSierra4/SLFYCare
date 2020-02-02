import numpy as np
import pandas as pd
import os
import datetime
import itertools
from scipy import stats

# load in data

apple_watch_dir = './healthdata'
hrv = pd.read_csv(os.path.join(apple_watch_dir, "HeartRateVariabilitySDNN.csv"))
sleep = pd.read_csv(os.path.join(apple_watch_dir, "SleepAnalysis.csv"))
sleep = sleep.loc[sleep["value"] == "HKCategoryValueSleepAnalysisAsleep"]
workouts = pd.read_csv(os.path.join(apple_watch_dir, "Workout.csv"))

today = datetime.datetime.today().date()
get_date = lambda x: '{}-{:02}-{:02}'.format(x.year, x.month, x.day)

days_back = 7

# convert dates in data to date objects
hrv['creationDate'] = pd.to_datetime(hrv['creationDate'])
# make compatible to compare with dates
for date_i in range(len(hrv)):
    if hrv['creationDate'].iloc[date_i].month < 10:
        month_str = "0" + str(hrv['creationDate'].iloc[date_i].month)
    else:
        month_str = str(hrv['creationDate'].iloc[date_i].month)
    if hrv['creationDate'].iloc[date_i].day < 10:
        day_str = "0" + str(hrv['creationDate'].iloc[date_i].day)
    else:
        day_str = str(hrv['creationDate'].iloc[date_i].day)
    hrv.at[date_i, 'date'] = str(hrv['creationDate'].iloc[date_i].year)+"-"+month_str+"-"+day_str


workouts['creationDate'] = pd.to_datetime(workouts['creationDate'])
for date_i in range(len(workouts)):
    if workouts['creationDate'].iloc[date_i].month < 10:
        month_str = "0"+str(workouts['creationDate'].iloc[date_i].month)
    else:
        month_str = str(workouts['creationDate'].iloc[date_i].month)
    if workouts['creationDate'].iloc[date_i].day < 10:
        day_str = "0"+str(workouts['creationDate'].iloc[date_i].day)
    else:
        day_str = str(workouts['creationDate'].iloc[date_i].day)
    workouts.at[date_i, 'date'] = str(workouts['creationDate'].iloc[date_i].year)+"-"+month_str+"-"+day_str

workouts['z_score_duration'] = stats.zscore(workouts['duration'])

sleep['creationDate'] = pd.to_datetime(sleep['creationDate'])
sleep['date'] = sleep['creationDate'].map(get_date)
sleep['startDate'] = pd.to_datetime(sleep['startDate'])
sleep['endDate'] = pd.to_datetime(sleep['endDate'])
sleep['timeDiff'] = (sleep['endDate'] - sleep['startDate'])

# generate the last week's dates

date_generator = (pd.Timestamp.today() - pd.Timedelta(days=i) for i in itertools.count())
last_week_dates = itertools.islice(date_generator, days_back)
last_week_dates = list(last_week_dates)

# convert to something that can compare
last_week_str = []
for last_week in last_week_dates:
    if last_week.month < 10:
        month_str = "0"+str(last_week.month)
    else:
        month_str = str(last_week.month)
    if last_week.day < 10:
        day_str = "0"+str(last_week.day)
    else:
        day_str = str(last_week.day)
    last_week_str.append(str(last_week.year)+"-"+month_str+"-"+day_str)


# calculate minutes of sleep a night

sleep_by_day = sleep.groupby('date')[["timeDiff"]].sum()
sleep_by_day = sleep_by_day.reset_index()

for date_i in range(len(sleep_by_day)):
    sleep_by_day.at[date_i,'hours'] = sleep_by_day.at[date_i, 'timeDiff'].total_seconds()/60/60
sleep_by_day['z_score'] = stats.zscore(sleep_by_day['hours'])

# average heart rate variability by day

hrv_by_day = hrv.groupby('date').mean()
hrv_by_day = hrv_by_day.reset_index()
hrv_by_day['z_score'] = stats.zscore(hrv_by_day['value'])

# calculate overall mean of measures

mean_hrv = hrv[["value"]].mean()
mean_workout_dur = workouts[["duration"]].mean()
mean_hr_sleep = sleep_by_day[["timeDiff"]].mean()
# select out the recent period dates from the data set

last_week_hrv = hrv_by_day.loc[hrv_by_day["date"].isin(last_week_str)]
last_week_workout = workouts[workouts["date"].isin(last_week_str)]
last_week_workout = last_week_workout[['date', 'duration', 'z_score_duration']]
last_week_hr_sleep = sleep_by_day[sleep_by_day["date"].isin(last_week_str)]



# get recent mean

#last_week_mean_hrv = last_week_hrv[["value"]].mean()
#last_week_mean_workout_dur = last_week_workout[["duration"]].mean()
#last_week_mean_hr_sleep = last_week_hr_sleep[["timeDiff"]].mean()

print(np.array(last_week_hrv['value']))
print(np.array(last_week_hr_sleep['hours']))
#print(last_week_mean_hr_sleep)
#print(np.array(last_week_hr_sleep['hours']) - last_week_mean_hr_sleep)

# print("Mean HRV (all): ", mean_hrv)
# print("Mean HRV (recent): ", last_week_mean_hrv)
#
# print("Mean Workout Duration (all): ", mean_workout_dur)
# print("Mean Workout Duration (recent): ", last_week_mean_workout_dur)
#
# print("Mean hours sleep (all): ", mean_hr_sleep)
# print("Mean hours sleep (recent): ", last_week_mean_hr_sleep)
