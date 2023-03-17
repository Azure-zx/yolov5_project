import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='path/to/default/source')
args = parser.parse_args()

input_source = args.source
print(input_source)  # 输出 'hello world'
