import re

path = '/home/piai/Documents/darknet/data'

with open(path+'/train.txt') as f:
    lines = f.readlines()

lines[0] # 'x64/Release/data/img/1.jpg\n'

p = re.compile('x64/Release/')
m = p.match(line[0])
print(m)


source = 'Lux, the Lady of Luminosity'
m = re.match('[a-z]+', source ) # 컴파일과 매치를 한번에 했다. 

# 0215 읽어들인 lines의 내용을 정규표현식을 사용해서 변경하려는 시도(펜딩)