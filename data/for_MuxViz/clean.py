
f = open("input.txt")

lines = f.readlines()

#print(lines[:2])

for i in lines:
    line = i.rstrip('\n')
    s = line.split(' ')
    to_write = s[2]+" "+s[1]+" 1\n"
    if(s[3]=='1'):
        fw = open("layer1.edges","a+")
        fw.write(to_write)
    if(s[4]=='1'):
        fw = open("layer2.edges","a+")
        fw.write(to_write)
    if(s[5]=='1'):
        fw = open("layer3.edges","a+")
        fw.write(to_write)
    if(s[6]=='1'):
        fw = open("layer4.edges","a+")
        fw.write(to_write)
    


