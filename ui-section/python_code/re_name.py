import sys
import os
re = sys.argv[1]
# vi = []
# re = "sdsdsd\sdsdsdsd\sdsd"

new_re = re.replace('\\','/')
# print("ori : "+re)
print(new_re)
# os.mkdir(new_re)
sys.stdout.flush()
