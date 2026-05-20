import kagglehub
import os

path = kagglehub.dataset_download(
    "subodhuniyal/malyalam-sarcasm"
)

print("Dataset Path:", path)

print("\nFiles:")
print(os.listdir(path))
