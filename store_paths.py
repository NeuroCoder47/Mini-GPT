#from pathlib import Path
#from shutil import copyfile

#script_path = Path(__file__).resolve()
#current_dir = script_path.parent

#all_path_list = [current_dir]
#print(current_dir)

#for sub_dir in current_dir.iterdir():
#    if sub_dir.is_dir():
#        print(sub_dir)
#        all_path_list.append(sub_dir)

#output_file = Path(current_dir) / "paths.txt"
#output_file.touch()

#with output_file.open("w", encoding="utf-8") as file:
#    for path in all_path_list:
#        print(path, file=file)

#print("Saved", output_file)

#target_folders = ["Preprocess Data (Pre Train)", "Tokenizer", "Train"]

#for folder_name in target_folders:
#    destination = current_dir / folder_name / "paths.txt"
#    copyfile(output_file, destination)
#    print("Copied to", destination)

# The thing is that this entire code will work But he doesnt ensure that the sequence in It'll store the path in the text file They will be same all the time
# Me be at one time the preprocess data file will be on the top And the tokenizer file will be Just below But in the other case, it might be the opposite.
# So this code below makes the entire thing extremely reproducible across even across different operating systems.
from pathlib import Path
from shutil import copyfile

script_path = Path(__file__).resolve()
current_dir = script_path.parent

target_folders = ["Preprocess Data (Pre Train)", "Tokenizer", "Train"]

all_path_list = [current_dir]
print(current_dir)

for folder_name in target_folders:
    folder_path = current_dir / folder_name
    print(folder_path)
    all_path_list.append(folder_path)

output_file = current_dir / "paths.txt"
output_file.touch()

with output_file.open("w", encoding="utf-8") as file:
    for path in all_path_list:
        print(path, file=file)

print("Saved", output_file)

for folder_name in target_folders:
    destination = current_dir / folder_name / output_file.name
    copyfile(output_file, destination)
    print("Copied to", destination)
