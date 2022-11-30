import os 
import datetime
import time
import shutil
import argparse
from down_loader_data import down_loader_data

def RemoveDir(filepath):
    ''' 
    如果文件夹不存在就创建，如果文件存在就清空！
    
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def create_git_order(time,real_out,history_out):

    """ 生成git指令并执行 """
    order_arr = ["git add {} {}".format(real_out,history_out),"git commit -m " + '"' + time + '"',"git push out_img master"] # 创建指令集合
    for order in order_arr:
        os.system(order) # 执行每一项指令

def put_file(time):
    """ 记录文件 """
    file = open(r"./git_push_time.txt","a+") 
    file.write(time + "完成一次提交\n")

def calculate_class_acc(out_list):
    """
    计算一个样本的各类概率
    :param out_label:
    :param out_list:
    :return:
    """

    label = ["N","C","M","X"]
    class_acc = {}

    for index in range(4):
        if index == 0:
            class_acc[label[index]] = len([i for i  in out_list if i < 100**(index+1)])/len(out_list)
        else:
            class_acc[label[index]] = len([i for i  in out_list if 10*10**index < i < 10*10**(index+1)])/len(out_list)

    class_out = max(class_acc,key=class_acc.get)

    print("out_label:",class_out,"out_mean:",np.array(out_list).mean(),"out_acc:",class_acc)



def record_put_file(str):
    """ 记录文件 """
    file = open(r"./sun_flares_record.txt","a+") 
    file.write(str+"\n")

def put_file(str):
    """ 记录文件 """
    file = open(r"./sun_flares.txt","w") 
    file.write(str+"\n")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Real_Time_Sun_Flares_Predict ')

    parser.add_argument('--time_gap', type=str, default="1h", help='')
    parser.add_argument('--time_length', type=int, default=6, help='')
    parser.add_argument('--realTime_data', type=str, default="./realTime_data/", help='')
    parser.add_argument('--historyTime_out', type=str, default="./historyTime_out/", help='')
    parser.add_argument('--realTime_out', type=str, default="./realTime_out/", help='')

    opt = parser.parse_args()

    if os.system("cd /home/dell460/ljm/Sun_flares/analy_output/5.real_time_pro/"):
        raise Exception("cd invalid run") 

    all_real_time = [0]

    for index in range(1000):

 
        # ############### 下载离当前时刻最近的数据，并保存到本地文件夹 ###############

        now_time = datetime.datetime.now().strftime("%Y.%m.%d_%H:%s_TAI")

        use_label,start_time, end_time,harp_num =  down_loader_data(opt.time_length,opt.time_gap,opt.realTime_data)  

        if  (use_label and (end_time!=all_real_time[-1])):

            print("begining test ------------")
            all_real_time.append(end_time)

            record_put_file("\n\nharp_num:{}".format(harp_num))
            record_put_file("\n\ntime:{}_{}".format(start_time, end_time))

            put_file("\n\nharp_num:{}\n\ntime:{}_{}".format(harp_num,start_time, end_time))

            # # ################# 运行网络测试程序 ###############
            if os.system("python real_time_test.py --data_path={} --save_path={} >text.log".format(opt.realTime_data, opt.realTime_out)):
                raise Exception("Test invalid run") 

            # 将当前结果备份到历史文件夹中
            if os.system("cp -r {} {}".format(opt.realTime_out+"/"+os.listdir(opt.realTime_out)[-1],opt.historyTime_out)):
                raise Exception("copy data invalid run") 


            # # ############### 将处理的结果、文件 传到github上 ############### 

            # 创建远程链接器
            try:
                os.system("git remote add out_img https://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@github.com/JiamengLv/Sun_Flares_Predict.git")
            except:
                raise Exception("git romote invalid run") 

            # 上传到github上
            create_git_order(now_time,opt.realTime_out,opt.historyTime_out)

            # 上传到slack上
            if os.system("python up_test.py"):
                raise Exception("up_slack invalid run") 

            # print("--------------------- push success ----------------------")

            # 清空real_time_out/data的结果和数据
            RemoveDir(opt.realTime_out)
            RemoveDir(opt.realTime_data)


        else:
            print("--No data available ------------")
            time.sleep(12*3600)
            continue
            
        # 延时
        time.sleep(12*3600)

        # print(now_time)
        # # ==》 从别的路径 copy 数据
        # if os.system("cp -r {} {}".format(real_time_data_path,data_path)):
        #     raise Exception("downloader data invalid run") 