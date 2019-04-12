import os
import os.path as osp
import sys

print("Path at terminal when executing this file")
print(os.getcwd() + "\n")

print("This file path, relative to os.getcwd()")
print(__file__ + "\n")

print("This file full path (following symlinks)")
full_path = osp.realpath(__file__)
print(full_path + "\n")

print("This file directory and name")
path, filename = osp.split(full_path)
print(path + ' --> ' + filename + "\n")

print("This file directory only")
print(osp.dirname(full_path)+'\n')

fn='../data'
print("This file full path (following symlinks)")
full_path = osp.realpath(fn)
print(full_path + "\n")
