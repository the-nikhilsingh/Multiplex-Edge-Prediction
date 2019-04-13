f = open("layout.txt","a+")

for i in range(1,301):
    x = str(i)
    f.write(x+","+x+"\n")