path=r'C:\Users\vedak\Music\Cyberbullying_Project\src\api.py'
with open(path,'r',encoding='utf-8') as f:
    lines = f.readlines()
for idx,line in enumerate(lines):
    if '# Startup/Shutdown' in line:
        print('found at', idx+1)
        for j in range(idx, idx+10):
            print(j+1, repr(lines[j]))
        break
