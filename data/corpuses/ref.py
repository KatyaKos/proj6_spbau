import re
import os

fin = open('hobbit.txt', 'r')
fout = open('new_hobbit.txt', 'w+')

string = ""
pattern = re.compile(r'\s+')
for line in fin:
    string += line
string = string.strip().lower().replace('\n', ' ').replace('\t', ' ')
string = re.sub(pattern, ' ', string)
fout.write(string)
fout.close()
