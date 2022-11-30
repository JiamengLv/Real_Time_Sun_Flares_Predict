from sunpy.net import Fido
from sunpy.net import attrs as a
import pandas as pd
from datetime import datetime as dt_obj
from datetime import timedelta
import datetime

# 将Time从字符串转换为datetime对象
def parse_tai_string(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])
    return dt_obj(year, month, day, hour, minute)

# 将Time从字符串转换为数字
def time_num(tstr):
    year = int(tstr[:4])
    month = int(tstr[5:7])
    day = int(tstr[8:10])
    hour = int(tstr[11:13])
    minute = int(tstr[14:16])

    number = year * 10 ** 8 + month * 10 ** 6 + day * 10 ** 4 + hour * 10 ** 2 + minute * 10 ** 2
    return number

# 开始时间，截止时间
tstart = "2022/07/01"
tend = "2022/10/05"



event_type = "FL"  # FL 耀斑
result = Fido.search(a.Time(tstart, tend),
                     a.hek.EventType(event_type),
                     a.hek.FL.GOESCls > "C1.0",
                     a.hek.OBS.Observatory == "GOES")

hek_results = result["hek"]
noaa_flares_info = hek_results["fl_goescls", "event_starttime", "event_peaktime", "event_endtime", "ar_noaanum"]

print(noaa_flares_info)

# 改变时间的保存形式
Peaktimes = []
for i in range(len(noaa_flares_info)):
    Peaktimes.append(parse_tai_string(str(noaa_flares_info['event_peaktime'][i])))
noaa_flares_info['event_peaktime'] = Peaktimes

event_starttime = []
for i in range(len(noaa_flares_info)):
    event_starttime.append(parse_tai_string(str(noaa_flares_info['event_starttime'][i])))
noaa_flares_info['event_starttime'] = event_starttime

event_endtime = []
for i in range(len(noaa_flares_info)):
    event_endtime.append(parse_tai_string(str(noaa_flares_info['event_endtime'][i])))
noaa_flares_info['event_endtime'] = event_endtime

# ar_noaanum ---> harpnum
# 加载 NOAA -- HARP 相对应的数据表
noaa_harp = pd.read_csv('http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt', sep=' ')

noaa_harp.to_csv("./1.csv")

n_listofactiveregions = list(noaa_flares_info['ar_noaanum'].flatten())
n_listofgoesclasses = list(noaa_flares_info['fl_goescls'].flatten())
list_eventpeaktime = list(noaa_flares_info['event_peaktime'].flatten())

classification = []

for i in range(len(n_listofactiveregions)):

    idx = noaa_harp[noaa_harp['NOAA_ARS'].str.contains(
            str(int(n_listofactiveregions[i])))]

    # 如果没有匹配的HARPNUM则退出
    if (idx.empty == True):
        print('skip: there are no matching HARPNUMs for',
              str(int(n_listofactiveregions[i])))
        continue

    single_class_instance = [idx.HARPNUM.values[0], str(int(n_listofactiveregions[i])), n_listofgoesclasses[i], str(list_eventpeaktime[i])]

    classification.append(single_class_instance)

harp_nums = {}
peak_times = []
flares_classes = []

for active_info in classification:
    harp_start_end = {}
    harp_num = str(active_info[0])

    if harp_num in harp_nums:

        harp_nums[harp_num]["peak_times"].append(active_info[3])
        harp_nums[harp_num]["flares_classes"].append(active_info[2])

        if time_num(str(harp_nums[harp_num]["start"])) < time_num(active_info[3]):
            harp_nums[harp_num]["start"] = harp_nums[harp_num]["start"]
        else:
            harp_nums[harp_num]["start"] = active_info[3]

        if time_num(harp_nums[harp_num]["end"]) > time_num(active_info[3]):
            harp_nums[harp_num]["end"] = harp_nums[harp_num]["end"]
        else:
            harp_nums[harp_num]["end"] = active_info[3]
    else:
        harp_nums[harp_num] = {}
        harp_nums[harp_num]["peak_times"] = [active_info[3]]
        harp_nums[harp_num]["flares_classes"] = [active_info[2]]
        harp_nums[harp_num]["start"] = active_info[3]
        harp_nums[harp_num]["end"] = active_info[3]

flares_classes = []

action_num = "8574"
start_time = "2022.09.04_17:00_TAI"

action = harp_nums["8574"]
endtime = parse_tai_string(start_time) + datetime.timedelta(hours=30)

try:
    for i in range(len(action["peak_times"])):
        if  parse_tai_string(start_time) < parse_tai_string(action["peak_times"][i]) and parse_tai_string(action["peak_times"][i])<endtime:
            flares_classes.append(action["flares_classes"][i])
            print("发生耀斑")
    if flares_classes==[]:
        print("不发生耀斑")
except:
    print("There is no {} data".format(action_num))


