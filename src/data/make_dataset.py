
import pandas as pd
from glob import glob

# Read single CSV file

single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv")

# List all data in data/raw/MetaMotion

files = glob("../../data/raw/MetaMotion/*")
len(files)

# Extract features from filename

data_path = "../../data/raw/MetaMotion\\" 

f = files[1] # Just for example take first file

participant = f.split("-")[0].lstrip(data_path)
label = f.split("-")[1]
category = f.split("-")[2].rstrip("123")

df = pd.read_csv(f)
df["participant"] = participant
df["label"] = label
df["category"] = category

# Read all files

data_path = "../../data/raw/MetaMotion\\"

gyr_df = pd.DataFrame()
acc_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for f in files:
    participant = f.split("-")[0].lstrip(data_path)
    label = f.split("-")[1]
    category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

    df = pd.read_csv(f)
    df["label"] = label
    df["participant"] = participant
    df["category"] = category

    if "Accelerometer" in f:
        df["Set"] = acc_set
        acc_df = pd.concat([acc_df, df])
        acc_set += 1

    elif "Gyroscope" in f:
        df["Set"] = gyr_set
        gyr_df = pd.concat([gyr_df, df])
        gyr_set += 1

acc_df.shape # An example from the merged dataset
gyr_df.shape # An example from the merged dataset

# Working with datetimes

acc_df.info() # Here epoch is an object variable
gyr_df.info() # Here epoch is an object variable

acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

acc_df.info() # Now epoch has been turned into a datetime variable which can be used later for calculation purposes
gyr_df.info() # Now epoch has been turned into a datetime variable which can be used later for calculation purposes

# Now we can drop all other time stamps as epoch will do it
acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis =1)
gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis =1)

# Turn into function

# Everything we did above was to figure out what all we need to do. Now we can 
# compile all of those into one single function

import pandas as pd
from glob import glob

def read_data_from_files(files):

    files = glob("../../data/raw/MetaMotion/*")

    data_path = "../../data/raw/MetaMotion\\"

    gyr_df = pd.DataFrame()
    acc_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].lstrip(data_path)
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")

        df = pd.read_csv(f)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["Set"] = acc_set
            acc_df = pd.concat([acc_df, df])
            acc_set += 1

        elif "Gyroscope" in f:
            df["Set"] = gyr_set
            gyr_df = pd.concat([gyr_df, df])
            gyr_set += 1
    
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    acc_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis =1, inplace=True)
    gyr_df.drop(["epoch (ms)", "time (01:00)", "elapsed (s)"], axis =1, inplace=True)

    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files(files)

# Merging datasets

data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df])

# Rename columns for easy understanding
data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set"
    ]

# Resample data (frequency conversion)

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "participant": "last",
    "label": "last",
    "category": "last",
    "set": "last"
    }

data_merged[ :1000].resample(rule="200ms").apply(sampling)
# But if we apply this sampling on the entire dataset then it will produce
# gigantic amounts of data, which will also have a lot of NaN values. This is
# why we'll split the data into days

# Split the data by day
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
data_resampled.info()

# Column "set" should be int not float
data_resampled["set"] = data_resampled["set"].astype("int")
data_resampled.info()

# Export dataset

data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")