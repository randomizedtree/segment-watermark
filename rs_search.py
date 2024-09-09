RC=0.6
Rt=0.15
code=[() for i in range(100)]
for b in range(12,36,4):
    for m in range(2,9):
        for k in range(1,2**m):
            for n in range(k+2,(2**m)+1):
                if (n-k)%2==1:
                    continue
                t=(n-k)/2
                if m*k!=b:
                    continue
                if k/n<RC:
                    continue
                if t/n<Rt:
                    continue
                if code[b]==():
                    code[b]=(n,k,m)
                else:
                    if code[b][0]>n:
                        code[b]=(n,k,m)

    print(b,":!!!",code[b])

