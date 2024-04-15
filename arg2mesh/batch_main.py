import arg2mesh
import glob
import os
# arg2mesh.arg2mesh("E:\\arg2mesh\\test\\output\\","E:\\arg2mesh\\test\\param_4147.txt","E:\\arg2mesh\\test\\param_inter_lines_4147.json")
pwd = os.getcwd()
input_dir = 'test'
input_path = os.path.join(pwd,input_dir)
input_file = os.path.join(input_path,'todo.txt')
with open(input_file,'r') as file:
    for line in file:
        param = glob.glob(os.path.join(input_path,line.strip(),'*.txt'))[0]
        inter_lines = glob.glob(os.path.join(input_path,line.strip(),'*.json'))[0]
        arg2mesh.arg2mesh(os.path.join(input_path,line.strip())+'\\',param,inter_lines)