root = "CNN_synth_testset"

#训练集（真假人脸各选了1000张）
f = open(root+"/train.txt","w")
for i in range(1000):
    name = "0"*(4-len(str(i)))+str(i)+".png"
    f.write("/0_real/"+name+" 0\n")
for i in range(1000):
    name = "0"*(4-len(str(i)))+str(i)+".png"
    if i !=999:#最后一行不加换行符
        f.write("/1_fake/"+name+" 1\n")
    else:
        f.write("/1_fake/"+name+" 1")
f.close()


#测试集(真假人脸各选了500张)
f = open(root+"/test.txt","w")
for i in range(1000,1999):
    name = "0"*(4-len(str(i)))+str(i)+".png"
    f.write("/0_real/"+name+" 0\n")
for i in range(1000,1999):
    name = "0"*(4-len(str(i)))+str(i)+".png"
    if i !=1998:#最后一行不加换行符
        f.write("/1_fake/"+name+" 1\n")
    else:
        f.write("/1_fake/"+name+" 1")
f.close()
