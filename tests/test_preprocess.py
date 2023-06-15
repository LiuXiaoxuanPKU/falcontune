import transformers
import torch
from datasets import load_dataset
from falcontune.data import TrainShareGPT, IGNORE_TOKEN_ID
from falcontune.model.falcon.model import RWConfig

def load_data(filename, ids):
    dataset = load_dataset("json", data_files=filename)
    return dataset.filter(lambda r: r["id"] in ids)

def load_tokenizer(name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    tokenizer.truncation_side = 'left'

    tokenizer.bos_token_id = None
    tokenizer.eos_token_id = tokenizer.vocab["<|endoftext|>"]
    tokenizer.pad_token_id = tokenizer.vocab["<|endoftext|>"]
    return tokenizer
    
def test_short():
    tokenizer = load_tokenizer("tiiuae/falcon-7b")
    sharegpt = TrainShareGPT("sharegpt", 0, tokenizer, 50)
    # datafile = "/rscratch/zhendong/lily/data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
    datafile = "tests/data.json"
    record_ids = ["IUWyjK9_0", "1", "2"]
    data = load_data(datafile, record_ids)["train"]
    processed = data.map(
        lambda ele: sharegpt.tokenize_inputs(ele),
        batched=True,
        load_from_cache_file=False
    ).filter(
        lambda ele: ele["raw_conversations"] != "IGNORE"
    )
    for raw, label in zip(processed["raw_conversations"], processed["labels"]):
        print(raw)
        print("=============================")
        label = torch.tensor(label)
        label = torch.where(label == IGNORE_TOKEN_ID, tokenizer.pad_token_id, label)
        print(tokenizer.decode(label))
        print("--------------------")

if __name__ == "__main__":
    test_short()