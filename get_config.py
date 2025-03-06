import pandas as pd
import json
df = pd.read_csv("./datasets/real/response.csv",header=None)
# counts = {
#                 'students_num':df['student_id'].nunique(),
#                 'exercises_num':df['exercise_id'].nunique(),
#             }
# with open('config.json','w')as config:
#     json.dump(counts,config)



last_column = df.iloc[:, 2]
new_column = []

for value in last_column:
    new_column.append(int(value)-1)

df.iloc[:, 2] = new_column

# 保存修改后的 CSV 文件
df.to_csv('datasets/real/response.csv', index=False)