#!/usr/bin/python3

import sys
import os
import time
import argparse
import json
from pathlib import Path
import pandas as pd
from natsort import natsorted

# each dataset contains a tuple of (ros2 data path, ground truth path)
OpenLORIS_dataset = [
    ("/dataset/SLAM/OpenLORIS/cafe1-2/ros2",
     "/dataset/SLAM/OpenLORIS/cafe1-2/non-ros/groundtruth.txt"),
    ("/dataset/SLAM/OpenLORIS/corridor1-1/ros2",
     "/dataset/SLAM/OpenLORIS/corridor1-1/non-ros/groundtruth.txt"),
    ("/dataset/SLAM/OpenLORIS/corridor1-2/ros2",
     "/dataset/SLAM/OpenLORIS/corridor1-2/non-ros/groundtruth.txt"),
    ("/dataset/SLAM/OpenLORIS/corridor1-3/ros2",
     "/dataset/SLAM/OpenLORIS/corridor1-3/non-ros/groundtruth.txt"),
    ("/dataset/SLAM/OpenLORIS/corridor1-4/ros2",
     "/dataset/SLAM/OpenLORIS/corridor1-4/non-ros/groundtruth.txt"),
    ("/dataset/SLAM/OpenLORIS/corridor1-5/ros2",
     "/dataset/SLAM/OpenLORIS/corridor1-5/non-ros/groundtruth.txt")
]
Euroc_dataset = [
    ("/dataset/SLAM/EuRoc/V1_01_easy/ros2",
     "/dataset/SLAM/EuRoc/V1_01_easy/mav0/state_groundtruth_estimate0/data.tum"),
    ("/dataset/SLAM/EuRoc/V1_03_difficult/ros2",
     "/dataset/SLAM/EuRoc/V1_03_difficult/mav0/state_groundtruth_estimate0/data.tum"),
    ("/dataset/SLAM/EuRoc/MH_01_easy/ros2",
     "/dataset/SLAM/EuRoc/MH_01_easy/mav0/state_groundtruth_estimate0/data.tum"),
    ("/dataset/SLAM/EuRoc/MH_04_difficult/ros2",
     "/dataset/SLAM/EuRoc/MH_04_difficult/mav0/state_groundtruth_estimate0/data.tum")
]


def DoCommand(cmd: str):
    print(
        'Running command: {}'.format(cmd))
    os.system(cmd)


def ParseArg():
    parser = argparse.ArgumentParser(description=(
        'Run the system and evaluate results. '
        'Usage example: python3 batch_run.py euroc ./TestResults --test_name my_test'),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        'dataset', help='dataset name, euroc or openloris', type=str)
    parser.add_argument('running_result_path',
                        help='path to the system running results', type=str)
    parser.add_argument('--test_name', help='test name', type=str)

    args = parser.parse_args()
    return args


def main():
    args = ParseArg()
    if args.dataset == 'euroc':
        dataset = Euroc_dataset
    elif args.dataset == 'openloris':
        dataset = OpenLORIS_dataset
    else:
        print('dataset name invalid !')
        sys.exit(1)

    base_result_path = os.path.abspath(args.running_result_path)
    test_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
    test_name = ''
    if args.test_name is not None:
        test_name = args.test_name
    test_folder_name = test_time + '_' + test_name
    for item in dataset:
        # run SLAM on the dataset
        data = item[0]
        ground_truth = item[1]
        shell_command = 'nohup ros2 launch delta_vins launch.py DataSourcePath:={} &'.format(
            data)
        DoCommand(shell_command)

        # wait for running process to finish
        log_file = os.path.expanduser('~/.ros/log/latest/launch.log')
        log_file_size = 0
        while True:
            time.sleep(10)
            # check if log file size is increasing
            if os.path.exists(log_file):
                new_file_size = os.path.getsize(log_file)
                if new_file_size > log_file_size:
                    log_file_size = new_file_size
                else:
                    print('log file size stop increasing, file size: {}'.format(
                        log_file_size))
                    shell_command = 'pkill -f "delta_vins"'
                    DoCommand(shell_command)
                    break
            else:
                print('log file {} not found !'.format(log_file))
                shell_command = 'pkill -f "delta_vins"'
                DoCommand(shell_command)
                sys.exit(1)

        # collect running results
        result_path = os.path.join(
            base_result_path, args.dataset, test_folder_name, os.path.basename(os.path.dirname(data)))
        shell_command = 'mkdir -p {}'.format(result_path)
        DoCommand(shell_command)
        shell_command = 'mv {} {}'.format(os.path.join(
            base_result_path, 'outputPose.tum'), result_path)
        DoCommand(shell_command)
        real_log_file = Path(log_file).resolve()
        shell_command = 'cp {} {}'.format(real_log_file, result_path)
        DoCommand(shell_command)

        # evaluate results
        shell_command = 'evo_rpe tum {} {} -d 15 -u f -r trans_part --save_results {}'.format(
            ground_truth, os.path.join(result_path, 'outputPose.tum'), os.path.join(result_path, 'rpe_trans.zip'))
        DoCommand(shell_command)
        shell_command = 'unzip -p {} {} > {}'.format(
            os.path.join(result_path, 'rpe_trans.zip'), 'stats.json', os.path.join(result_path, "rpe_trans.json"))
        DoCommand(shell_command)
        shell_command = 'evo_rpe tum {} {} -d 15 -u f -r angle_deg --save_results {}'.format(
            ground_truth, os.path.join(result_path, 'outputPose.tum'), os.path.join(result_path, 'rpe_rot.zip'))
        DoCommand(shell_command)
        shell_command = 'unzip -p {} {} > {}'.format(
            os.path.join(result_path, 'rpe_rot.zip'), 'stats.json', os.path.join(result_path, "rpe_rot.json"))
        DoCommand(shell_command)

    # collect evaluation results
    evaluation_list = []
    for root, dirs, files in os.walk(os.path.join(
            base_result_path, args.dataset, test_folder_name)):
        if 'rpe_trans.json' in files and 'rpe_rot.json' in files:
            evaluation = {}
            with open(os.path.join(root, 'rpe_trans.json'), 'r') as f:
                contents = json.load(f)
                evaluation["trans_rmse(m)"] = contents["rmse"]
                evaluation["trans_max(m)"] = contents["max"]
                evaluation["trans_min(m)"] = contents["min"]
            with open(os.path.join(root, 'rpe_rot.json'), 'r') as f:
                contents = json.load(f)
                evaluation["rot_rmse(deg)"] = contents["rmse"]
                evaluation["rot_max(deg)"] = contents["max"]
                evaluation["rot_min(deg)"] = contents["min"]
            evaluation_list.append((os.path.basename(root), evaluation))
    evaluation_list = natsorted(evaluation_list, key=lambda x: x[0])
    evaluation_list = [{"Dataset": name, **info}
                       for name, info in evaluation_list]
    df = pd.DataFrame(evaluation_list)
    print('\ntest:', test_name, '\n', df)

    with open(os.path.join(
            base_result_path, args.dataset, test_folder_name, "evaluations.csv"), "w", encoding="utf-8") as f:
        test_name = ''
        if args.test_name is not None:
            test_name = args.test_name
        f.write("Test: {} \n".format(test_name))
        df.to_csv(f, index=False)


if __name__ == '__main__':
    main()
