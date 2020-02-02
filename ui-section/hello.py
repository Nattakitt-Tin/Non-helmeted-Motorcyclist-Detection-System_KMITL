import sys

re = sys.argv[1]

# re = "sdsdsd\sdsdsdsd\sdsd"

new_re = re.replace('\\','/')
print("ori : "+re)
print("new : "+new_re)
sys.stdout.flush()
