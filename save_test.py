
# # # Define an empty list
# # train = []

# # # Open the file and read the content in a list
# # with open('train.txt', 'r') as filehandle:
# #     for line in filehandle:
# #         # Remove linebreak which is the last character of the string
# #         curr_place = line
# #         # Add item to the list
# #         train.append(curr_place)

# # print(train)
# print(train[0])
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
import csv
 
f = open('val.txt', mode='r') # 打开txt文件，以‘utf-8'编码读取
lines = f.readlines()  # 以行的形式进行读取文件
# line = f.read()
# list1 = []
# a = line.split()
# a = line.strip()

paths = []
for i in range(len(lines)):
    paths.append(lines[i][48:].strip())


# with open('train_withoutHeader.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     for string in paths:
#         writer.writerow([string])

with open('val_withoutHeader.txt', 'w') as file:
    for string in paths:
        file.write(string + '\n')




# print(line)

