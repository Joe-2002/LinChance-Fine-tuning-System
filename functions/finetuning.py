import re
import json
# 参考：
# https://blog.csdn.net/advance1989/article/details/131334855?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-131334855-blog-104373682.235^v43^pc_blog_bottom_relevance_base4&spm=1001.2101.3001.4242.1&utm_relevant_index=3

# 4. 数据集大小：对于较小的数据集，使用较小的Batch_size可能会导致模型欠拟合，而较大的Batch_size可能会导致模型过拟合。
# 一般来说，可以从小到大尝试不同的Batch_size，观察训练过程中的loss变化和模型性能，选择使得loss下降稳定且模型性能最佳的Batch_size。此外，也可以根据经验选择常用的Batch_size，如32、64、128等。

# 当Batch_size增大时，每个step需要处理更多的样本，在同样的时间内完成一个epoch的训练次数会减少，从而导致训练速度变慢。这是因为较大的Batch_size需要更多的计算资源和内存空间，而且在处理大量数据时也需要更多的时间。
# 此外，较大的Batch_size可能会导致模型在训练过程中陷入局部最优解，并且可能会导致模型泛化能力下降。因此，在选择Batch_size时需要在训练速度和模型性能之间做出权衡。

#     --learning_rate 1.8079646145791248e-05 \
#     --num_train_epochs 5 \
def determine_batch_size(model_name):
    # 使用正则表达式提取模型大小的信息
    match = re.search(r'(\d+)B', model_name)
    
    if match:
        model_size = int(match.group(1))
        
        # 根据模型大小来选择合适的batch_size
        if model_size <= 2:
            return 1  # 选择 batch_size 为 1，即在线学习
        elif model_size <= 7:
            return 2  # 根据实际情况选择适当的 batch_size
        else:
            return 3  # 根据实际情况选择更大的 batch_size
    # 如果没有匹配到模型大小信息，默认返回一个适当的 batch_size
    return 2

def read_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def determine_learning_rate(num_samples):
    if num_samples < 1000:
        return 1e-3
    elif 1000 <= num_samples < 10000:
        return 3e-5
    else:
        return 5e-5
def determine_num_train_epochs(num_samples):
    if num_samples < 1000:
        return 5
    elif 1000 <= num_samples < 10000:
        return 4
    else:
        return 3



        
