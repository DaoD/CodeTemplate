import argparse
import random
import numpy as np
import torch
import logging
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from model import MyModel
from transformers import get_linear_schedule_with_warmup, BertTokenizer
from evaluate import Metrics
from myDataset import FileDataset
from tqdm import tqdm
import os

task_config_dict = {
    'douban': {
        'max_uttr_num': 10,
        'max_uttr_len': 50,
        'max_response_len' : 50,
        'max_sequence_len': 256
    }
}

parser = argparse.ArgumentParser()
parser.add_argument("--is_training",
                    default=True,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--per_gpu_batch_size",
                    default=24,
                    type=int,
                    help="The batch size.")
parser.add_argument("--per_gpu_test_batch_size",
                    default=16,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=5e-4,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--task",
                    default="douban",
                    type=str,
                    help="Task")
parser.add_argument("--epochs",
                    default=15,
                    type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./model/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--log_path",
                    default="./log/",
                    type=str,
                    help="The path to save log.")

args = parser.parse_args()
args.batch_size = args.per_gpu_batch_size * torch.cuda.device_count()
args.test_batch_size = args.per_gpu_test_batch_size * torch.cuda.device_count()
result_path = "./output/" + args.task + "/"
args.save_path += args.task + "." + MyModel.__name__ + ".pt"
args.score_file_path = result_path + MyModel.__name__ + "." + args.score_file_path
args.log_path += args.task + "." + MyModel.__name__ + ".log"

device = torch.device("cuda:0")
print(args)
logging.basicConfig(filename=args.log_path, level=logging.DEBUG)  # debug level - avoid printing log information during training
logger = logging.getLogger(__name__)
logger.info(args)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
additional_tokens = 1
tokenizer.add_tokens("[eos]")

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def train_model():
    train_data = "./data/" + args.task + "/train.txt"
    valid_data = "./data/" + args.task + "/dev.txt"
    model = MyModel()
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info('* number of parameters: %d' % n_params)  # compute the number of parameters
    model = model.to(device)  # move the model to GPU
    # model = torch.nn.DataParallel(model)  # open this if using multi-GPU for training
    fit(model, train_data, valid_data)

def train_step(model, train_data, criterion):
    with torch.no_grad():
        for key in train_data.keys():
            train_data[key] = train_data[key].to(device)
    y_pred = model.forward(train_data)
    batch_y = train_data["labels"]
    loss = criterion(y_pred, batch_y)
    return loss

def fit(model, X_train, X_dev):
    train_dataset = FileDataset(X_train, task_config_dict[args.task]['max_uttr_len'], task_config_dict[args.task]['max_response_len'], task_config_dict[args.task]['max_sequence_len'], tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)  # set num_workers > 1 for speeding up
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    t_total = int(len(train_dataset) * args.epochs // args.batch_size)
    one_epoch_step = len(train_dataset) // args.batch_size
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(one_epoch_step * 0.1), num_training_steps=t_total)
    criterion = torch.nn.BCEWithLogitsLoss()
    one_epoch_step = len(train_dataset) // args.batch_size
    scheduler = get_linear_schedule_with_warmup  # decay the learning rate
    best_result = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for epoch in range(args.epochs):
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        avg_loss = 0.0
        model.train()
        epoch_iterator = tqdm(train_dataloader, ncols=120)  # set ncols for the length of the progress bar
        for i, training_data in enumerate(epoch_iterator):
            loss = train_step(model, training_data, criterion)
            loss = loss.mean()
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.learning_rate = param_group['lr']
            epoch_iterator.set_postfix(lr=args.learning_rate, loss=loss.item())  # show the learning rate and loss on the progress bar
            if i > 0 and i % (one_epoch_step // 5) == 0:  # evaluate every 20% data
                best_result = evaluate(model, X_dev, best_result)
                model.train()
            avg_loss += loss.item()
        best_result = evaluate(model, X_dev, best_result)
        model.train()
        cnt = len(train_dataset) // args.batch_size + 1
        tqdm.write("Average loss:{:.6f}".format(avg_loss / cnt))

def evaluate(model, X_dev, best_result, is_test=False):
    y_pred, y_label = predict(model, X_dev)
    metrics = Metrics(args.score_file_path)
    with open(args.score_file_path, 'w') as output:
        for score, label in zip(y_pred, y_label):
            output.write(str(score) + '\t' + str(label) + '\n')
    result = metrics.evaluate_all_metrics()
    if not is_test and sum(result) > sum(best_result):
        best_result = result
        tqdm.write("Best Result: MAP: %.4f MRR: %.4f P@1: %.4f R10@1: %.4f R10@2: %.4f R10@5: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        logger.info("Best Result: MAP: %.4f MRR: %.4f P@1: %.4f R10@1: %.4f R10@2: %.4f R10@5: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), args.save_path)
    if is_test:
        print("Best Result: MAP: %.4f MRR: %.4f P@1: %.4f R10@1: %.4f R10@2: %.4f R10@5: %.4f" % (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
    return best_result

def predict(model, X_dev):
    model.eval()
    test_dataset = FileDataset(X_dev, task_config_dict[args.task]['max_uttr_len'], task_config_dict[args.task]['max_response_len'], task_config_dict[args.task]['max_sequence_len'], tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataloader, ncols=130, leave=False)
        for i, test_data in enumerate(epoch_iterator):
            with torch.no_grad():
                for key in test_data.keys():
                    test_data[key] = test_data[key].to(device)
            y_pred_batch = model.forward(test_data)
            y_pred.append(y_pred_batch.data.cpu().numpy().reshape(-1))
            y_label_batch = test_data["labels"].data.cpu().numpy().tolist()
            y_label.append(y_label_batch)
    y_pred = np.concatenate(y_pred, axis=0).tolist()
    y_label = np.concatenate(y_label, axis=0).tolist()
    return y_pred, y_label

def test_model():
    test_data = "./data/" + args.task + "/text.txt" 
    word_embeddings = torch.FloatTensor(torch.load("./data/" + args.task + "/word_embeddings.pt"))  # load pre-trained embeddings
    model = MyModel(word_embeddings)
    model_state_dict = torch.load(args.save_path)
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    evaluate(model, test_data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], True)

if __name__ == '__main__':
    set_seed(0)
    if args.is_training:
        train_model()
        test_model()
    else:
        test_model()
