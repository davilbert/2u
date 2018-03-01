def transcribe(dep):
    for i in range(m):
        if dep[i]=='T':
            dep[i] = 'U'
    return dep
t = []
m=int(input())
for i in range(m):
    t.append(input())
print(transcribe(t))
def reverse(s):
    h = list(s)
    h.reverse()
    return ''.join(h)
#print(reverse('CCGGAAGAGCTTACTTAG'))
l = []
n=int(input())
for i in range(n):
        l.append(input())
print(l)
def trans(l):
    l2=[]
    for i in range(n):
        if l[i]== 'A':
            l2.append('T')
        if l[i]== 'T':
            l2.append('A')
        if l[i]== 'G':
            l2.append('C')
        if l[i]== 'C':
            l2.append('G')
    return l2
print(trans(l))

