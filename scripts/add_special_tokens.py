from transformers import AutoTokenizer
import argparse
import string

def get_tokens_to_add():
    tokens_to_add = list()
    # for i in range(1024):
    #     tokens_to_add.append(f'<{i}>')
    tokens_to_add += ['<caption_task>', '<calculation_task>', '<prove_task>']
    tokens_to_add += ['<prompt>', '</prompt>', '<caption>', '</caption>', 
        '<layout>', '</layout>', '<text>', '</text>', '<stru>', '</stru>', '<sem>', '</sem>']
    tokens_to_add += ['>>', '<plugin>', '</plugin>', '<ans>', '</ans>', '<type>', '</type>']
    return tokens_to_add


def get_tokens_to_add_cn(tokens_to_add=None):
    if not tokens_to_add: tokens_to_add = []
    tokens_to_add += list(string.ascii_letters)
    tokens_to_add += ['△', '∥', '∵', '∴', '⊥', '\\\\frac', '≌', '☉', '’', 'Ð', 
        '⁀', '▱', '丄', '∘', 'ɑ', '㎝', '…', '➝', '⇔', '¬']
    tokens_to_add += ['<prompt>', '</prompt>', '<ocr>', '</ocr>', 
        '<layout>', '</layout>', '<text>', '</text>', '<detect>', '</detect>']
    tokens_to_add += ['>>', '<plugin>', '</plugin>', '<ans>', '</ans>', '<type>', '</type>']
    return tokens_to_add

def modify_vocab_bert(source_path, target_path):
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    tokens_to_add = get_tokens_to_add()
    print(len(tokenizer))
    num_add_tokens = tokenizer.add_tokens(tokens_to_add)
    print(len(tokenizer))
    print(num_add_tokens)

    tokenizer.save_pretrained(target_path)


def modify_vocab_bert_cn(source_path, target_path):
    tokenizer = AutoTokenizer.from_pretrained(source_path)
    tokens_to_add = get_tokens_to_add_cn()
    print(len(tokenizer))
    num_add_tokens = tokenizer.add_tokens(tokens_to_add)
    print(len(tokenizer))
    print(num_add_tokens)

    tokenizer.save_pretrained(target_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', type=str, default='')
    parser.add_argument('--target_path', type=str, default='')
    args = parser.parse_args()
    modify_vocab_bert(
        args.source_path,
        args.target_path
    )