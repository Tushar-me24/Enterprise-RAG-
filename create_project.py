from pathlib import Path

#Current WOrking Directory
root = Path(".")

#Folders to create
folders = [
    "app/api",
    "app/core",
    "app/rag",
    "app/services",
    "data",
    "templates",
    "static",
    "uploads",
    "tests",
]


#Files to create
files= [
    "app/main.py",
    "requirments.txt",
    "ingest_sample_kb.py",
    "run.py",
    ".env"
]


#Create Folder
for folder in folders:
    (root / folder).mkdir(parents=True , exist_ok=True)

#Create File
for file in files:
    (root / file).touch(exist_ok=True)



print("Project Structure Created Successfully")