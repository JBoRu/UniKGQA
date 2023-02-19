'''
	> File Name: test.py
	> Author: tuzhuo
	> Mail: xmb028@163.com
	> Created Time: Tue 15 Sep 2020 10:07:25 AM CST
'''

'''
这个代码能做到多进程并行
'''

import sys
import time
import subprocess
import numpy as np
import torch
from manager import GPUManager
import random
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i', dest='cmd_file', default=None)
parser.add_option('-t', dest='waiting_seconds', default=None)
parser.add_option('-n', dest='max_online_task', default=6, type=int)
parser.add_option('-o', dest='sh_output', default='./multi_gpu_runner.record.log', type=str)
parser.add_option('-p', dest='priority', default="['0','1','2','3','4','5','6','7']", type=str)
parser.add_option('-m', dest='memory_free', default=20000, type=int)
(options, args) = parser.parse_args()

'''
cmd_1 = 'CUDA_VISIBLE_DEVICES=' + str(gpu_index) + ' ' + 'python /public/home/ztu/projects/basenji-old/bin/basenji_sat_bed2.py -f /public/home/WBXie/tuzhuo/tzhuo/genome/NIPP.fasta -l 400 --rc -t /public/home/ztu/projects/nxiepo/2/yuanjian/bw54-list.txt -o /public/home/ztu/projects/nxiepo/lwh_predict/lbd/h5_result -n lbd /public/home/ztu/projects/nxiepo/2/training/ZS97_params.txt /public/home/ztu/projects/nxiepo/2/training/r-zs-54/model_best.tf /public/home/ztu/projects/nxiepo/lwh_predict/lbd/lbd.bed'
'''
### 从命令行读取所有指令 ###
mission_queue = []

if (options.cmd_file != None):
    cmds =  open(options.cmd_file)
    line = cmds.readline()
    while(line):
        line = line.strip()
        if ('#' not in line) and ('python' in line):#仅支持python语言的GPU计算任务
            mission_queue.append(line)
        line = cmds.readline()
    cmds.close()

if (len(mission_queue) <= 0):
    sys.exit(0)

print('Got %d Missions\n'%(len(mission_queue)))
sh_fo = open(options.sh_output, "w")
print('Log output to %s\n'%(sh_fo))

### GPU 管理对象 ###
gm=GPUManager(eval(options.priority), options.memory_free)

p = [] # 存储正在gpu上执行的程序进程
total = len(mission_queue)
finished = 0
running = 0
count = 0
allocated_gpu_ac = []
while(finished < total):
    if finished + running == total:
        time.sleep(60)

    localtime = time.asctime( time.localtime(time.time()) )
    gpu_av = gm.choose_no_task_gpu() # 可用的gpu indexs
    gpu_av = list(set(gpu_av)-set(allocated_gpu_ac))

    # 在每轮epoch当中仅提交1个GPU计算任务
    if len(gpu_av) > 0 and running < options.max_online_task and len(mission_queue) > 0:
        gpu_index = random.sample(gpu_av, 1)[0]#为了保证服务器上所有GPU负载均衡，从所有空闲GPU当中随机选择一个执行本轮次的计算任务
        task = mission_queue.pop(0)
        cmd_ = 'CUDA_VISIBLE_DEVICES=' + str(gpu_index) + ' ' + task # mission_queue当中的任务采用先进先出优先级策略
        print('Mission %d : %s\nRUN ON GPU : %d\nStarted @ %s\n'%(count, cmd_, gpu_index, localtime))
        count += 1
        # subprocess.call(cmd_, shell=True)
        p.append({task: subprocess.Popen(cmd_, shell=True, stdout=sh_fo, stderr=sh_fo), 'gpu_idx':gpu_index, 'task_idx':count-1})
        running += 1
        allocated_gpu_ac.append(gpu_index)
        time.sleep(int(options.waiting_seconds)) #等待NVIDIA CUDA代码库初始化并启动

    else:#如果服务器上所有GPU都已经满载则不提交GPU计算任务
        print('Keep Looking @ %s'%(localtime), end = '\r')
        time.sleep(20)

    new_p = []#用来存储已经提交到GPU但是还没结束计算的进程
    for i in range(len(p)):
        # if list(p[i].values())[0].poll() != None:
        #     running -= 1
        #     finished += 1
        #     gi = p[i]['gpu_idx']
        #     allocated_gpu_ac = list(set(allocated_gpu_ac)-set([gi]))
        # else:
        #     new_p.append(p[i])
        if list(p[i].values())[0].poll() == None:
            new_p.append(p[i])
        # elif list(p[i].values())[0].poll() == 100: # CUDA 超显存退出重回任务队列
        #     running -= 1
        #     task_restart = list(p[i].keys())[0]
        #     mission_queue.append(task_restart)
        #     gi = p[i]['gpu_idx']
        #     ti = p[i]['task_idx']
        #     allocated_gpu_ac = list(set(allocated_gpu_ac) - set([gi]))
        #     print("Mission %d on GPU %d : %s\n Out of CUDA memory, reallocate in mission queue!"%(ti,gi,task_restart))
        # elif list(p[i].values())[0].poll() == 101:  # 手动断掉显卡占用
        #     print("Mission %d on GPU %d : %s\n is killed manually!"%(ti,gi,task_restart))
        #     time.sleep(60)
        #     running -= 1
        #     task_restart = list(p[i].keys())[0]
        #     mission_queue.append(task_restart)
        #     gi = p[i]['gpu_idx']
        #     ti = p[i]['task_idx']
        #     allocated_gpu_ac = list(set(allocated_gpu_ac) - set([gi]))
        else: # 其它情况退出均结束程序
            running -= 1
            task_restart = list(p[i].keys())[0]
            mission_queue.append(task_restart)
            gi = p[i]['gpu_idx']
            ti = p[i]['task_idx']
            allocated_gpu_ac = list(set(allocated_gpu_ac) - set([gi]))
            print("Mission %d on GPU %d : %s\n is killed manually!" % (ti, gi, task_restart))
            time.sleep(60)
        # elif list(p[i].values())[0].poll() <= 0:#子进程被杀掉或者正常退出
        #     running -= 1
        #     finished += 1
        #     gi = p[i]['gpu_idx']
        #     allocated_gpu_ac = list(set(allocated_gpu_ac)-set([gi]))
        # elif list(p[i].values())[0].poll() > 0: #子进程异常退出，returncode对应于出错码；
        #     running -= 1
        #     task_restart = list(p[i].keys())[0]
        #     mission_queue.append(task_restart)

    if len(new_p) == len(p):#此时说明已提交GPU的进程队列当中没有进程被执行完
        time.sleep(5)
    p = new_p

# print('All jobs submited ! Waiting %d jobs finished ! '%(len(p)))
print('All jobs finished !')
for i in range(len(p)):#mission_queue队列当中的所有GPU计算任务均已提交，等待GPU计算完毕结束主进程
    list(p[i].values())[0].wait()

print('%d Mission Complete ! Checking GPU Process Over ! '%(finished+len(p)) )
sh_fo.close()
