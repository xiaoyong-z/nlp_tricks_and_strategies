
input_dath='./output/sentiment_300.txt'
output_path='./vocab/sentiment_300'

# 读文件
def read_from_txt(txt):
    with open(txt, 'r') as fr:
        html = fr.read()
    return html
# 写文件
def save_to_txt(txt, result):
    with open(txt, 'a') as fw:
        fw.write(result)
# 按行读文件
def read_line(txt):
    with open(txt, 'r')as fr:
        list = fr.readlines()
    return list

a=read_line(input_dath)
print(len(a))
print(a[0],a[231521],a[-1])
save_to_txt(output_path,str(len(a))+' '+str(300)+'\n')
for i in range(0,len(a)):
    save_to_txt(output_path,str(a[i]))
