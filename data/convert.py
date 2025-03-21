import tensorflow as tf
import pandas as pd
import os
import glob
import re

# Path to folder containing CSVs
csv_folder = r"C:\Users\19083\Desktop\Luigi-s-Mansion\data"  # Replace with your actual folder path

# Get all CSV files, sorted numerically
csv_files = [f for f in glob.glob(os.path.join(csv_folder, "*_mario_dqn_log.csv")) 
             if re.search(r'(\d+)_mario_dqn_log\.csv$', f)]
if csv_files == None:
    print('No Files Found')
    quit()
else:
    print(f"Collect: {csv_files}")

# Combine all CSV files into one DataFrame
df_list = [pd.read_csv(file) for file in csv_files]
combined_df = pd.concat(df_list, ignore_index=True)

# Optional: Save the merged CSV
# combined_csv_path = os.path.join(csv_folder, "combined_mario_dqn_log.csv")
# combined_df.to_csv(combined_csv_path, index=False)

# Keep only numeric columns
df_numeric = combined_df.select_dtypes(include=['number'])

# Create TensorBoard logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a SummaryWriter
writer = tf.summary.create_file_writer(log_dir)

# Write the merged data to TensorBoard
with writer.as_default():
    for step, row in df_numeric.iterrows():
        for col in df_numeric.columns:
            tf.summary.scalar(col, float(row[col]), step=step)

# Close the writer
writer.close()

# print(f"Combined CSV saved at: {combined_csv_path}")
print("Run the following command to view TensorBoard:")
print("tensorboard --logdir=logs --port=6007")