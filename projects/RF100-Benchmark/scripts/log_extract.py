import re
import argparse
import os
import csv
import json
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment

parser = argparse.ArgumentParser(description="log_name")
# parser.add_argument("result", type='str', help="present result address")
# parser.add_argument("dataset", type=str, help="present dataset result address")
# parser.add_argument("--result_csv", type=str, default=None, required = False, help='flag of csv')
parser.add_argument("method", type=str, help='method name')
parser.add_argument("epoch", type=int, help = 'train_epoch')
parser.add_argument('--work_dirs', type=str, default='work_dirs/', required=False, help='save directory')
args = parser.parse_args()

def main():
    num = 0
    fail_num = 0
    none_exist_num = 0
    fail=[]
    none_exist=[]
    for dataset in sorted(os.listdir("rf100/")):
        # dataset = args.dataset.split('/')[-1]
        print(f'\ndataset={dataset}, index={num}')
        # import pdb;pdb.set_trace()
        # determine whether the dataset directory exists
        try:
            dirs = [args.work_dirs+dataset+'/'+d for d in os.listdir(args.work_dirs+dataset) if os.path.isdir(args.work_dirs+dataset+'/'+d)]
            num+=1
        except:
            print(f"{dataset} directory doesn't exist!")
            none_exist_num+=1
            none_exist.append(dataset)
            continue
        dirs.sort(key=os.path.getmtime)
        latest_dir = dirs[-1]
        latest_log_name = latest_dir.split('/')[-1]
        print('time='+latest_log_name)

        latest_log = latest_dir+f'/{latest_log_name}.log'
        print(latest_log)

        with open(latest_log, 'r') as f:
            log = f.read()
        with open('rf100/'+dataset+'/train/_annotations.coco.json','r') as f:
            image=json.load(f)
            num_train = len(image['images'])        
        with open('rf100/'+dataset+'/valid/_annotations.coco.json','r') as f:
            image=json.load(f)
            num_valid = len(image['images'])    
        with open('scripts/labels_names.json') as f:
            label=json.load(f)
            for index in label:
                if index['name'] == dataset:
                    category = index['category']
        complete_flag=re.findall(r'Epoch\(val\) \[{}\]\[\d+/\d+\]'.format(args.epoch), log)
        # Determine whether the training is complete
        if not complete_flag:
            fail_num+=1
            fail.append(dataset)
            print("------------------------------------------------------------------------------------")
            print(f'{dataset} train failed!')
            print(f'{fail_num} dataset failed!')
            print("------------------------------------------------------------------------------------")
            key_value=[dataset, category, num_train, num_valid, '', '', '', '', '']
            if num==1:
                    result_csv = args.work_dirs+f'{latest_log_name}_final_eval.csv'
                    print(result_csv)
                    with open(result_csv, mode='a') as f:
                        writer = csv.writer(f)
                        header1 = ['Dataset', 'Category', 'Images', 'Images', args.method, args.method, args.method, args.method, args.method]
                        writer.writerow(header1)
                        header2 = ['', '', 'train', 'valid', 'mAP', 'mAP50', 'mAP_s', 'mAP_m', 'mAP_l']
                        writer.writerow(header2)
                        writer.writerow(key_value)

            else:
                with open(result_csv, mode='a') as f:
                    writer = csv.writer(f)
                    writer.writerow(key_value)
        else:
            print(f'com={complete_flag}')
            '''match result'''
            match = re.findall(r'The best checkpoint with ([\d.]+) coco/bbox_mAP at ([\d.]+) epoch', log)[-1]
            best_epoch = match[-1]
            print(f'best_epoch={best_epoch}')
            match_AP = re.findall(r'\[{}\]\[\d+/\d+\]    coco/bbox_mAP: (-?\d+\.?\d*)  coco/bbox_mAP_50: (-?\d+\.?\d*)  coco/bbox_mAP_75: -?\d+\.?\d*  coco/bbox_mAP_s: (-?\d+\.?\d*)  coco/bbox_mAP_m: (-?\d+\.?\d*)  coco/bbox_mAP_l: (-?\d+\.?\d*)'.format(best_epoch), log)
            print(f'match_AP={match_AP}')
            
            # value = match.group()
            # print(value)
            # print(match)
            # best=match[-1]
            # print(best)
            key_value = [dataset, category, num_train, num_valid]
            key_value.extend(match_AP[0])
            
            # key_value.extend(float(x) for x in match_AP[0])
            if num==1:
                result_csv = args.work_dirs+f'{latest_log_name}_final_eval.csv'
                print(result_csv)
                with open(result_csv, mode='a') as f:
                    writer = csv.writer(f)
                    header1 = ['Dataset', 'Category', 'Images', 'Images', args.method, args.method, args.method, args.method, args.method]
                    writer.writerow(header1)
                    header2 = ['', '', 'train', 'valid', 'mAP', 'mAP50', 'mAP_s', 'mAP_m', 'mAP_l']
                    writer.writerow(header2)
                    writer.writerow(key_value)
                print(f'result is saved in {result_csv}')
            else:
                with open(result_csv, mode='a') as f:
                    writer = csv.writer(f)
                    writer.writerow(key_value)

    # Convert .csv file to .xlsx file
    df = pd.read_csv(result_csv)
    result_xlsx = '{}.xlsx'.format(result_csv.split('.')[0])
    print(f'\n{result_xlsx} created!\n')
    df.to_excel(result_xlsx)
    wb = load_workbook(result_xlsx)
    ws = wb.active
    ws.merge_cells('D1:E1')
    ws.merge_cells('F1:J1')
    ws['C1'].alignment = Alignment(horizontal='center', vertical='center')
    ws['E1'].alignment = Alignment(horizontal='center', vertical='center')
    wb.save(result_xlsx)
    print(f'{none_exist_num} datasets were not trained:\n{none_exist}\n')
    print(f'{fail_num} training failed:\n{fail}')
            # try:
            #     result_csv
            #     print('create result_csv')
            # except NameError:
            #     print('result_csv exists')

if __name__ == "__main__":
    main()
     