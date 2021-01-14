import os
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_size", type=int, default=12000)
parser.add_argument("--limit_alphabet", type=int, default=3000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False,  # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##",
)

special_tokens = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]"
]  # 스페셜 토큰
special_tokens += [f"[UNK{i}]" for i in range(10)]
special_tokens += [f"[unused{i}]" for i in range(100)]

tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size,
    min_frequency=10,
    special_tokens=special_tokens,
)

tokenizer.save("ch-{}-wpm-{}.json".format(args.limit_alphabet, args.vocab_size), True)
