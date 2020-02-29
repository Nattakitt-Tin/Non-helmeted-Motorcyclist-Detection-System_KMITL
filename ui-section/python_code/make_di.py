import os
import sys
di = sys.argv[1]
new_di = di.replace('\\','/')
fo1 = "/extracted"
new_di = new_di+fo1
os.mkdir(new_di)
os.mkdir(new_di+"/no_helmet")
os.mkdir(new_di+"/helmet")
print(new_di)
sys.stdout.flush()
