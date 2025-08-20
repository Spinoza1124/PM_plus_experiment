import train_ser
from train_ser import parse_arguments
import sys
import pickle
import os
import time
import numpy as np
from collections import defaultdict


repeat_kfold = 2 # 执行n次重复的LOSO交叉验证
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

#------------PARAMETERS---------------#

features_file = '/mnt/shareEEx/liuyang/code/PM_plus_experiment/data/IEMOCAP_multi/IEMOCAP_multi.pkl'

# LOSO (Leave-One-Speaker-Out) 交叉验证设置
# 每次选择一个说话人作为测试集，其余作为训练集
all_speakers = ['1M','1F','2M','2F','3M','3F','4M','4F','5M','5F']

# 训练参数
num_epochs  = '100'
early_stop = '8'
batch_size  = '64'
lr          = '0.00001'
random_seed = 111
gpu = '1'
gpu_ids = ['0']
save_label = str_time

# 创建结果保存目录
results_dir = f'results_LOSO_{save_label}'
os.makedirs(results_dir, exist_ok=True)
models_dir = os.path.join(results_dir, 'models')
os.makedirs(models_dir, exist_ok=True)
 

# 开始LOSO交叉验证
all_stat = []
fold_results = defaultdict(list)  # 存储每个fold的详细结果

print(f"开始LOSO交叉验证，共{repeat_kfold}次重复，每次{len(all_speakers)}个fold")
print(f"结果将保存到: {results_dir}")
print("="*60)

for repeat in range(repeat_kfold):
    print(f"\n重复 {repeat+1}/{repeat_kfold}")
    print("-"*40)
    
    random_seed += (repeat*100)
    seed = str(random_seed)
    
    repeat_stats = []
    
    # LOSO: 每次留出一个说话人作为测试集
    for fold_idx, test_speaker in enumerate(all_speakers):
        print(f"\nFold {fold_idx+1}/{len(all_speakers)}: 测试说话人 = {test_speaker}")
        
        # 从剩余说话人中选择一个作为验证集，其余作为训练集
        remaining_speakers = [s for s in all_speakers if s != test_speaker]
        val_speaker = remaining_speakers[fold_idx % len(remaining_speakers)]  # 循环选择验证说话人
        
        print(f"验证说话人 = {val_speaker}")
        print(f"训练说话人 = {[s for s in remaining_speakers if s != val_speaker]}")
        
        # 设置当前fold的模型保存路径
        current_save_label = f"{save_label}_repeat{repeat}_fold{fold_idx}_test{test_speaker}"
        model_save_path = os.path.join(models_dir, f"{current_save_label}.pth")
        
        # 设置训练参数
        train_ser.sys.argv = [
            'train_ser.py', 
            features_file,
            '--repeat_idx', str(repeat),
            '--val_id', val_speaker, 
            '--test_id', test_speaker,
            '--gpu', gpu,
            '--gpu_ids', gpu_ids,
            '--num_epochs', num_epochs,
            '--early_stop', early_stop,
            '--batch_size', batch_size,
            '--lr', lr,
            '--seed', seed,
            '--save_label', current_save_label,
            '--pretrained'
        ]
        
        # 执行训练和测试
        stat = train_ser.main(parse_arguments(sys.argv[1:]))
        
        # 保存模型权重到指定目录
        if os.path.exists(current_save_label + '.pth'):
            os.rename(current_save_label + '.pth', model_save_path)
            print(f"模型已保存到: {model_save_path}")
        
        # 记录结果
        fold_result = {
            'repeat': repeat,
            'fold': fold_idx,
            'test_speaker': test_speaker,
            'val_speaker': val_speaker,
            'stats': stat,
            'model_path': model_save_path
        }
        
        all_stat.append(stat)
        repeat_stats.append(fold_result)
        fold_results[f'repeat_{repeat}'].append(fold_result)
        
        # 打印当前fold结果
        print(f"Fold {fold_idx+1} 结果: 最佳epoch={stat[0]}, 总epoch={stat[1]}, "
              f"测试损失={stat[8]}, 测试WA={stat[9]}, 测试UA={stat[10]}")
    
    # 保存当前重复的结果
    repeat_result_file = os.path.join(results_dir, f'repeat_{repeat}_results.pkl')
    with open(repeat_result_file, "wb") as fout:
        pickle.dump(repeat_stats, fout)
    print(f"\n重复 {repeat+1} 结果已保存到: {repeat_result_file}")

# 计算最终统计结果
n_total = repeat_kfold * len(all_speakers)
total_best_epoch, total_epoch, total_loss, total_wa, total_ua = 0, 0, 0, 0, 0

print("\n" + "="*60)
print("LOSO交叉验证详细结果:")
print("="*60)

# 打印每个fold的详细结果
for i in range(n_total):
    repeat_idx = i // len(all_speakers)
    fold_idx = i % len(all_speakers)
    test_speaker = all_speakers[fold_idx]
    
    print(f"重复{repeat_idx+1} Fold{fold_idx+1} (测试:{test_speaker}): "
          f"最佳epoch={all_stat[i][0]}, 总epoch={all_stat[i][1]}, "
          f"损失={all_stat[i][8]}, WA={all_stat[i][9]}, UA={all_stat[i][10]}")
    
    total_best_epoch += all_stat[i][0]
    total_epoch += all_stat[i][1]
    total_loss += float(all_stat[i][8])
    total_wa += float(all_stat[i][9])
    total_ua += float(all_stat[i][10])

# 计算平均结果
avg_best_epoch = total_best_epoch / n_total
avg_epoch = total_epoch / n_total
avg_loss = total_loss / n_total
avg_wa = total_wa / n_total
avg_ua = total_ua / n_total

print("\n" + "="*60)
print("LOSO交叉验证最终结果:")
print("="*60)
print(f"平均最佳epoch: {avg_best_epoch:.2f}")
print(f"平均总epoch: {avg_epoch:.2f}")
print(f"平均测试损失: {avg_loss:.4f}")
print(f"平均加权准确率(WA): {avg_wa:.2f}%")
print(f"平均非加权准确率(UA): {avg_ua:.2f}%")

# 计算标准差
wa_values = [float(stat[9]) for stat in all_stat]
ua_values = [float(stat[10]) for stat in all_stat]
loss_values = [float(stat[8]) for stat in all_stat]

wa_std = np.std(wa_values)
ua_std = np.std(ua_values)
loss_std = np.std(loss_values)

print(f"\n标准差:")
print(f"WA标准差: {wa_std:.2f}%")
print(f"UA标准差: {ua_std:.2f}%")
print(f"损失标准差: {loss_std:.4f}")

# 保存完整的实验结果
final_results = {
    'experiment_info': {
        'timestamp': str_time,
        'repeat_kfold': repeat_kfold,
        'num_speakers': len(all_speakers),
        'total_folds': n_total,
        'features_file': features_file,
        'hyperparameters': {
            'num_epochs': num_epochs,
            'early_stop': early_stop,
            'batch_size': batch_size,
            'lr': lr,
            'initial_seed': 111
        }
    },
    'detailed_results': fold_results,
    'summary_statistics': {
        'avg_best_epoch': avg_best_epoch,
        'avg_epoch': avg_epoch,
        'avg_loss': avg_loss,
        'avg_wa': avg_wa,
        'avg_ua': avg_ua,
        'wa_std': wa_std,
        'ua_std': ua_std,
        'loss_std': loss_std,
        'all_wa_values': wa_values,
        'all_ua_values': ua_values,
        'all_loss_values': loss_values
    },
    'raw_stats': all_stat
}

# 保存最终结果
final_results_file = os.path.join(results_dir, 'final_LOSO_results.pkl')
with open(final_results_file, "wb") as fout:
    pickle.dump(final_results, fout)

# 保存CSV格式的结果摘要
import csv
csv_file = os.path.join(results_dir, 'LOSO_results_summary.csv')
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['重复', 'Fold', '测试说话人', '验证说话人', '最佳Epoch', '总Epoch', '测试损失', '测试WA', '测试UA'])
    
    for repeat_key, repeat_data in fold_results.items():
        for fold_data in repeat_data:
            writer.writerow([
                fold_data['repeat'] + 1,
                fold_data['fold'] + 1,
                fold_data['test_speaker'],
                fold_data['val_speaker'],
                fold_data['stats'][0],
                fold_data['stats'][1],
                fold_data['stats'][8],
                fold_data['stats'][9],
                fold_data['stats'][10]
            ])
    
    # 添加平均结果行
    writer.writerow(['平均', '', '', '', f'{avg_best_epoch:.2f}', f'{avg_epoch:.2f}', 
                    f'{avg_loss:.4f}', f'{avg_wa:.2f}', f'{avg_ua:.2f}'])
    writer.writerow(['标准差', '', '', '', '', '', f'{loss_std:.4f}', f'{wa_std:.2f}', f'{ua_std:.2f}'])

print(f"\n实验结果已保存:")
print(f"- 完整结果: {final_results_file}")
print(f"- CSV摘要: {csv_file}")
print(f"- 模型权重目录: {models_dir}")
print(f"\nLOSO交叉验证完成!")
