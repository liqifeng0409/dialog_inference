import argparse
import csv
import re

import jieba
import torch
from tqdm import tqdm
from transformers import BertTokenizer, MT5ForConditionalGeneration
from torch.utils.data import DataLoader, Dataset
import numpy as np

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
import collections.abc as container_abcs

int_classes = int
string_classes = str


# from FakeMen.train_with_finetune import T5PegasusTokenizer


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


class T5PegasusTokenizer(BertTokenizer):
    """结合中文特点完善的Tokenizer
    基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
    """

    def __init__(self, pre_tokenizer=lambda x: jieba.cut(x, HMM=False), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_tokenizer = pre_tokenizer

    def _tokenize(self, text, *arg, **kwargs):
        split_tokens = []
        for text in self.pre_tokenizer(text):
            if text in self.vocab:
                split_tokens.append(text)
            else:
                split_tokens.extend(super()._tokenize(text))
        return split_tokens


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out).to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)

        return default_collate([default_collate(elem) for elem in batch])

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def create_data(data, tokenizer, args):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag, title = [], True, None
    for content in data:
        if type(content) == tuple:
            title, content = content
        if flag:
            flag = False
            print(content)
        if args.pretrain_model == 't5':
            text_ids = tokenizer.encode_plus(content, max_length=args.max_len,
                                             truncation='only_first', return_tensors='pt')

            features = {'input_ids': text_ids['input_ids'],
                        # 'attention_mask': [1] * len(text_ids),
                        'attention_mask': text_ids['attention_mask'],
                        'raw_data': content}
        elif args.pretrain_model == 'bart':
            in_ids = tokenizer.encode_plus(text=content, max_length=args.max_len, padding='max_length',
                                           truncation='only_first', return_tensors='pt')
            features = {'input_ids': in_ids['input_ids'].squeeze(0),
                        'attention_mask': in_ids['attention_mask'].squeeze(0),
                        'raw_data': content
                        }
        else:
            features = None
            print("NO DATA !")
        if title:
            features['title'] = title
        ret.append(features)
    return ret


def load_data(filename):
    """加载数据
    单条格式：(正文) 或 (标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            if len(l) > 600:
                continue
            cur = l.strip().split('\t')
            if len(cur) == 2:
                title, content = cur[0], cur[1]
                D.append((title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
    return D


class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def prepare_data(args, tokenizer):
    """准备batch数据
    """
    test_data = load_data(args.test_data)
    test_data = create_data(test_data, tokenizer, args)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_collate)
    return test_data


def generate2(test_data, model, tokenizer, args):
    gens, summaries = [], []
    with open(args.result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        model.eval()
        for feature in tqdm(test_data):
            raw_data = feature['raw_data']
            content = {k: v for k, v in feature.items() if k not in ['raw_data', 'title']}
            gen = model.generate(max_length=args.max_len_generate,
                                 eos_token_id=tokenizer.sep_token_id,
                                 decoder_start_token_id=tokenizer.cls_token_id,
                                 **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            writer.writerows(zip(gen, raw_data))
            gens.extend(gen)
            if 'title' in feature:
                summaries.extend(feature['title'])
    # if summaries:
    #     scores = compute_rouges(gens, summaries)
    #     print(scores)
    print('Done!')


def init_argument():
    parser = argparse.ArgumentParser(description='FakeMen')
    # parser.add_argument('--test_data', default='./data2/dev.tsv')
    parser.add_argument('--pretrain_model', default='bart', help='bart or t5')
    parser.add_argument('--model_dir', default='./saved_model')
    # parser.add_argument('--result_file', default='./data1/predict_result_t5.tsv')
    # parser.add_argument('--batch_size', default=16, help='batch size')
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=128, help='max length of outputs')
    parser.add_argument('--use_multiprocess', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_argument()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrain_model == 'bart':
        model = torch.load("./saved_model/bart_model_epoch_19.pt").to(device)  # 加载模型
        tokenizer = BertTokenizer.from_pretrained('./bart-base-chinese')
    elif args.pretrain_model == 't5':
        tokenizer = T5PegasusTokenizer.from_pretrained('../t5/t5_pegasus_pretrain/small')
        model = torch.load("./saved_model/t5_model_epoch_29_0.360.pt").to(device)
        # config5 = AutoConfig.from_pretrained('./t5_pegasus_pretrain/small')
        # model = Li(config5).to(device)
    else:
        model = None
        tokenizer = None
        print("no model and tokenizer !")

    bingli = "主诉：腹泻伴稀水样便。现病史：患儿无明显诱因下出现腹泻伴稀水样便情况，无咳嗽咳痰，无恶心呕吐，无其他明显不适症状。" \
             "精神状态一般，胃纳一般，余如常。辅助检查：暂无。既往史：既往体健。诊断：小儿腹泻。建议：予蒙脱石散服用，必要时就医查大便常规。"
    bingli2 = "主诉：发烧伴流涕1天。现病史：1天前患儿无明显诱因下出现发烧伴流涕情况，无咳嗽咳痰，无恶心呕吐，无其他明显不适症状。精神状态一般，" \
              "胃纳一般，余如常。辅助检查：暂无。既往史：既往体健。诊断：上呼吸道感染。建议：建议公立医院儿科治疗，密切观察。"
    bingli3 = "主诉：头痛3天。现病史：患者出现头痛3天，伴呕吐，一进食就吐，无发热，肌肉注射喜炎平治疗，精神差。辅助检查：无。既往史：无。" \
              "诊断：上呼吸道感染。建议：多喝水，可以口服吗叮林混悬液、利巴伟林颗粒和小儿氨酚黄那敏；医院就诊检查血常规。"
    bingli4 = "主诉：咳嗽。现病史：患儿干咳，鼻塞，流鼻涕。现服用盐酸氨溴索。辅助检查：既往史：过敏性咳嗽。诊断：咳嗽待查。建议：四季抗病毒口服液，" \
              "肺力咳，加强护理，勤喂水"
    bingli5 = "主诉：要求牙周治疗。现病史：近一个月，患者右上和右下的牙周炎较严重，肿的时候无法咀嚼食物，牙齿松动，要求检查。没有冷热刺激的情况。" \
              "患者睡觉磨牙导致牙龈出血。两年前曾在私人诊所洁治。每天早晚刷两次牙，每次三分钟。。既往史：否认新冠肺炎史和接触史，" \
              "否认近期发热史和疫区往返史。家族史：无特殊。全身：糖尿病：患病一年，服用二甲双胍，平时空腹血糖控制至8mmol/L；备孕期间注射胰岛素；" \
              "牙齿疼痛期间服用消炎药，阿莫西林和甲硝锉。"
    history = []
    shuru1 = '主诉：发热一周。现病史：患儿7天前出现发热，最高39℃，自服美林、百蕊颗粒、利巴韦林，效果可，2天前出现腹痛，脐周痛，呕吐，食欲差。' \
             '辅助检查：血常规。既往史：不详。诊断：上呼吸道感染。建议：继续当前治疗方案，复查血常规，监测体温，清淡饮食。[SEP]'
    model.eval()

    while True:
        text = input("医生:")
        history.append(text)
        shuru1 += text + '[SEP]'  # 此处选择病历 可在data/dev.tsv找更多病历信息
        xinxi = tokenizer.encode_plus(shuru1, return_tensors='pt', max_length=args.max_len)
        content = {
            'input_ids': xinxi['input_ids'].cuda(),
            'attention_mask': xinxi['attention_mask'].cuda()
        }

        gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             repetition_penalty=1.5,
                             **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gen = [item.replace(' ', '') for item in gen]
        shuru1 += ''.join(gen) + '[SEP]'
        print("患者:", ''.join(gen))
