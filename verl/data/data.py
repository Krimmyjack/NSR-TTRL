import pandas as pd
pd.set_option('display.max_colwidth', None)  # 不截断列宽
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.max_rows', None)     # 显示所有行（如果需要）
# 读取 Parquet 文件
file_path = "/root/autodl-tmp/TTRL/verl/data/MATH-TTT/answer.jsonl_0.4_pi1-32_r1.parquet"
df = pd.read_parquet(file_path)

# 查看前几行数据
print(df)

# 查看列名
print(df.columns)

# 查看数据的基本信息
print(df.info())