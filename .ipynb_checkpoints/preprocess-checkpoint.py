import os
import argparse
import sys
import subprocess
import glob
import shutil
import dvc.api

from collections import Counter
from git.repo.base import Repo
from smexperiments.tracker import Tracker
from torchvision.datasets.utils import extract_archive
from sklearn.model_selection import train_test_split
from pathlib import Path


dvc_repo_url = os.environ.get('DVC_REPO_URL')
dvc_branch = os.environ.get('DVC_BRANCH')

git_user = os.environ.get('GIT_USER', "sagemaker")
git_email = os.environ.get('GIT_EMAIL', "sagemaker-processing@example.com")

ml_root = Path("/opt/ml/processing")

dataset_zip = ml_root / "input" / "intel.zip"
git_path = ml_root / "sagemaker-intelimage"

def configure_git():
    subprocess.check_call(['git', 'config', '--global', 'user.email', f'"{git_email}"'])
    subprocess.check_call(['git', 'config', '--global', 'user.name', f'"{git_user}"'])
    
def clone_dvc_git_repo():
    print(f"\t:: Cloning repo: {dvc_repo_url}")
    
    repo = Repo.clone_from(dvc_repo_url, git_path.absolute(),allow_unsafe_protocols=True)
    
    return repo

def sync_data_with_dvc(repo):
    os.chdir(git_path)
    print(f":: Create branch {dvc_branch}")
    try:
        repo.git.checkout('-b', dvc_branch)
        print(f"\t:: Create a new branch: {dvc_branch}")
    except:
        repo.git.checkout(dvc_branch)
        print(f"\t:: Checkout existing branch: {dvc_branch}")
    print(":: Add files to DVC")
    
    subprocess.check_call(['dvc', 'add', "dataset"])
    
    repo.git.add(all=True)
    repo.git.commit('-m', f"'add data for {dvc_branch}'")
    
    print("\t:: Push data to DVC")
    subprocess.check_call(['dvc', 'push'])
    
    print("\t:: Push dvc metadata to git")
    repo.remote(name='origin')
    repo.git.push('--set-upstream', repo.remote().name, dvc_branch, '--force')

    sha = repo.head.commit.hexsha
    
    print(f":: Commit Hash: {sha}")

    with Tracker.load() as tracker:
        tracker.log_parameters({"data_commit_hash": sha})

def write_dataset(image_paths, output_dir):
    for img_path in image_paths:
        Path(output_dir / img_path.parent.stem).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(img_path, output_dir / img_path.parent.stem / img_path.name)

def generate_dataset():
    dataset_extracted = ml_root / "intel-image-classification"
    dataset_extracted.mkdir(parents=True, exist_ok=True)

    # split dataset and save to their directories
    print(f":: Extracting Zip {dataset_zip} to {dataset_extracted}")
    extract_archive(from_path=dataset_zip, to_path=dataset_extracted)

    ds = list((dataset_extracted / "seg_train" / "seg_train").glob("*/*"))
    ds += list((dataset_extracted / "seg_test" / "seg_test").glob("*/*"))
    d_pred = list((dataset_extracted / "seg_pred" / "seg_pred").glob("*/"))

    labels = [x.parent.stem for x in ds]
    print(":: Dataset Class Counts: ", Counter(labels))

    d_train, d_test = train_test_split(ds, test_size=0.2, stratify=labels)
    d_test, d_val = train_test_split(
        d_test, test_size=0.5, stratify=[x.parent.stem for x in d_test]
    )

    print("\t:: Train Dataset Class Counts: ", Counter(x.parent.stem for x in d_train))
    print("\t:: Test Dataset Class Counts: ", Counter(x.parent.stem for x in d_test))
    print("\t:: Val Dataset Class Counts: ", Counter(x.parent.stem for x in d_val))
    print("\t:: Total validation images", len(d_pred))

    for path in ["train", "test", "val"]:
        output_dir = git_path / "dataset" / path
        print(f"\t:: Creating Directory {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    print(":: Writing Datasets")
    write_dataset(d_train, git_path / "dataset" / "train")
    write_dataset(d_test, git_path / "dataset" / "test")
    write_dataset(d_val, git_path / "dataset" / "val")
    write_dataset(d_pred, git_path / "dataset" / "pred")        
                
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    # setup git
    print(":: Configuring Git")
    configure_git()
    
    print(":: Cloning Git")
    repo = clone_dvc_git_repo()
    
    print(":: Generate Dataset")
    # extract the input zip file and split into train and test
    generate_dataset()
    
    print(":: Sync Processed Data to Git & DVC")
    sync_data_with_dvc(repo)