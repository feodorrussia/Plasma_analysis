import shutil
import os
import time

init_path = "D:/Work/Data/dbs/"
destination_path = "C:/Users/f.belous/Work/Projects/Plasma_analysis/data/dbs/"

count = 0
for date_dir in os.listdir(init_path):
    if os.path.isfile(date_dir):
        continue
    for shot_dir in os.listdir(init_path + date_dir + "/"):
        if os.path.isfile(shot_dir):
            continue
        for file in os.listdir(init_path + date_dir + "/" + shot_dir + "/"):
            if os.path.isfile(init_path + date_dir + "/" + shot_dir + "/" + file):
                start_time = time.time()
                if os.path.splitext(file)[-1].lower() == ".sht" and not os.path.exists(destination_path + "sht/" + file):
                    shutil.copyfile(init_path + date_dir + "/" + shot_dir + "/" + file, destination_path + "sht/" + file)
                if os.path.splitext(file)[-1].lower() == ".dat" and not os.path.exists(destination_path + "dat/" + file):
                    shutil.copyfile(init_path + date_dir + "/" + shot_dir + "/" + file, destination_path + "dat/" + file)
                    
                if os.path.splitext(file)[-1].lower() == ".sht" or os.path.splitext(file)[-1].lower() == ".dat":
                    count += 1
                    print(count, "-", init_path + date_dir + "/" + shot_dir + "/" + file, end="")
                    print(f" - {os.path.getsize(init_path + date_dir + '/' + shot_dir + '/' + file) / 2 ** 20:.3f} MB", end="")
                    print(f" - Tooks: {(time.time() - start_time):.3f} ms")
