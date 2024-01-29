import ray
from ray import tune
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
import torch

def train_bert(config):
    # 加载数据集
    dataset = load_dataset('glue', 'mrpc')
    train_dataset = dataset['train']
    eval_dataset = dataset['validation_matched']

    # 加载模型和分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # 定义优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    # 训练模型
    for epoch in range(config['epochs']):
        for i in range(0, len(train_dataset), config['batch_size']):
            batch = train_dataset[i:i+config['batch_size']]
            inputs = tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True, return_tensors='pt')
            labels = torch.tensor(batch['label'])
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # 在验证集上评估模型性能
        eval_acc = evaluate(model, eval_dataset, tokenizer)
        tune.report(eval_acc=eval_acc)

def evaluate(model, dataset, tokenizer):
    correct = 0
    total = 0
    for i in range(0, len(dataset), 32):
        batch = dataset[i:i+32]
        inputs = tokenizer(batch['sentence1'], batch['sentence2'], padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(batch['label'])
        outputs = model(**inputs, labels=labels)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return correct / total

if __name__ == '__main__':
    ray.init()
    analysis = tune.run(
        train_bert,
        config={
            'lr': tune.loguniform(1e-5, 1e-3),
            'batch_size': tune.choice([16, 32, 64]),
            'epochs': 3,
            'step_size': tune.choice([1, 2, 4]),
            'gamma': tune.choice([0.1, 0.5, 0.9])
        },
        metric='eval_acc',
        mode='max',
        num_samples=10,
        resources_per_trial={'cpu': 2, 'gpu': 0.5},
        local_dir='./ray_results'
    )
    print('Best hyperparameters:', analysis.best_config)
