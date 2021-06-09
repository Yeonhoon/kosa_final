

import argparse

parse = argparse.ArgumentParser() # 인자값을 받을 수 있는 인스턴스 생성
parse.add_argument('--target', default='무엇을 요구하는가?', help="요구사항 쓰기")
parse.add_argument('--env', default="dev", help="실행환경이 뭐임?")
args= parse.parse_args()

print(args.target)

print(args.env)


import easydict

args2 = easydict.EasyDict({
    "width": 200,
    "height": 150,
    "length": 100
})

print(args2.width)
print(args2.height)