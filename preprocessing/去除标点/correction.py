import re,time,os
import pyltp
from pyltp import Segmentor
import jieba

def read_txt(txtPath):
    word_list = []
    with open(txtPath, 'r', encoding='utf-8') as readFile:
        for line in readFile:
            line = line.replace('\n','')
            word_list.append(line)

    readFile.close()
    return word_list

def save_txt(txtPath, writeLine):
    with open(txtPath, 'a', encoding='utf-8') as writeFile:
        writeFile.write(writeLine+'\n')

    writeFile.close()

def get_stopwords_list(stopwords_path):  
    stopwords_list = [line.strip() for line in open(stopwords_path, 'r', encoding='utf-8').readlines()]  
    return stopwords_list

def dealPuncations(content):
    # 中文
    content = content.replace('＃', ' ')
    content = content.replace('◤', ' ')
    content = content.replace('◢', ' ')
    content = content.replace('“', ' ')
    content = content.replace('”', ' ')
    content = content.replace('‘', ' ')
    content = content.replace('’', ' ')
    content = content.replace('【', ' ')
    content = content.replace('】', ' ')
    content = content.replace('［', ' ')
    content = content.replace('］', ' ')
    content = content.replace('「', ' ')
    content = content.replace('」', ' ')
    content = content.replace('〈', ' ')
    content = content.replace('〉', ' ')
    content = content.replace('（', ' ')
    content = content.replace('）', ' ')
    content = content.replace('《', ' ')
    content = content.replace('》', ' ')

    content = content.replace('；', '，')
    content = content.replace('、', '，')
    content = content.replace('：', '，')

    content = content.replace('…', '。')
    content = content.replace('……', '。')
    content = content.replace('？', '。')
    content = content.replace('！', '。')
    content = content.replace('～', '。')
    content = content.replace('/', '')
    content = content.replace('*', '')
    
    # 英文
    content = content.replace('<', ' ')
    content = content.replace('>', ' ')
    content = content.replace('[', ' ')
    content = content.replace(']', ' ')
    content = content.replace('{', ' ')
    content = content.replace('}', ' ')
    content = content.replace("'", ' ')
    content = content.replace('"', ' ')
    content = content.replace(',', ' ')
    content = content.replace('#', ' ')

    content = content.replace('*', '')
    content = content.replace('/', '')
    content = content.replace('(', '')
    content = content.replace(')', '')
    content = content.replace('?', '')
    content = content.replace('-', '')
    content = content.replace('—', '')
    content = content.replace('&', '')
    content = content.replace(':', '')
    content = content.replace('^', '')
    content = content.replace('.', '')
    content = content.replace('~', '。')
    content = content.replace('!', '。')
    content = content.replace('?', '。')

    # 连符号
    content = re.sub(r'\s+', '，', content)   #对所有空格转换为逗号
    content = re.sub(r'，{2,}', '。', content) #对多个逗号转换为单句号
    content = re.sub(r'。{2,}', '。', content) #对多个句号转换为单句号

    # 双符号替换
    content = content.replace('。，', '。')
    content = content.replace('，。', '。')

    return content

def deleteEmoji(content):
    # 参考：https://zhuanlan.zhihu.com/p/41213713
    try:
            # Wide UCS-4 build
            myre = re.compile(u'['
                              u'\U0001F300-\U0001F64F'
                              u'\U0001F680-\U0001F6FF'
                              u'\u2600-\u2B55'
                              u'\u23cf'
                              u'\u23e9'
                              u'\u231a'
                              u'\u3030'
                              u'\ufe0f'
                              u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u'\U00010000-\U0010ffff'
                               u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
                               u'\U00002702-\U000027B0]+',     
                              re.UNICODE)
    except re.error as E:
        # Narrow UCS-2 build
        myre =   re.compile(u'('
                                  u'\ud83c[\udf00-\udfff]|'
                                  u'\ud83d[\udc00-\ude4f]|'
                                  u'\uD83D[\uDE80-\uDEFF]|'
                                  u"(\ud83d[\ude00-\ude4f])|"  # emoticon
                                  u'[\u2600-\u2B55]|'
                                  u'[\u23cf]|'
                                  u'[\u1f918]|'
                                    u'[\u23e9]|'
                                  u'[\u231a]|'
                                  u'[\u3030]|'
                                  u'[\ufe0f]|'
                                  u'\uD83D[\uDE00-\uDE4F]|'
                                  u'\uD83C[\uDDE0-\uDDFF]|'
                                u'[\u2702-\u27B0]|'
                                  u'\uD83D[\uDC00-\uDDFF])+',
                                  re.UNICODE)

    content=myre.sub(' ', content)
    return content

def preparationForWordCut(content):
    #1- 处理标点符号
    content = dealPuncations(content)
    # print(content)
    # time.sleep(5)
    # save_txt(output_path,content)

    #2- 处理emoji符号
    content = deleteEmoji(content)
    # print(content)
    # time.sleep(5)
    # save_txt(output_path,content)

    return content

def word_segmentation(content):
    global ltp_wordSegmentModel_path, input_dish_path, input_location_path, input_dict_path,\
        stopwords_path
    dishName_list = read_txt(input_dish_path)
    locationName_list = read_txt(input_location_path)
    jieba.load_userdict(input_dict_path)    # 加载dish & location自定义词典
    stopwords_list = get_stopwords_list(stopwords_path)     # 停用词加载
    # wordSegmentor_ltp=Segmentor()
    # wordSegmentor_ltp.load(ltp_wordSegmentModel_path)
    #1- 先按句号分句
    content_sentence_list = content.split('。')

    #2- 遍历每一句，分句处理
    new_content = ''
    for content_sentence in content_sentence_list:
        #3- 按逗号分割，并遍历分割后语句
        new_content_sentence = ''
        content_colon_list = content_sentence.split('，')
        for content_colon in content_colon_list:
            #4- 开始分词-jieba
            words_segment_list = jieba.cut(content_colon)
            #5- 停用词过滤
            for word in words_segment_list:  
                if word not in stopwords_list:  
                    if (word.isprintable()):
                        new_content_sentence += (word + ' ')

        #6- 该句分词完成,保存至new_content
        new_content += new_content_sentence

    #print(new_content)
    return new_content

def main():
    #1- 按行读入all_text
    global input_text_path, output_text_path
    with open(input_text_path, 'r', encoding='utf-8') as readFile:
        for line in readFile:
            content = line.replace('\n','')[1:-1]   #注意：文本必须是被""包括住的，才能[1:-1]
            content = re.sub(r'\s+', '', content)

            #2- 文本预处理
            content = preparationForWordCut(content)

            #3- 分词
            content_seg = word_segmentation(content)

            #4- 保存
            save_txt(output_text_path,content_seg)


if __name__ == '__main__':
    # input: all_text => 官方测试集；all_text11 => 官方训练集 + 爬虫评论集
    project_path = os.getcwd()
    word_lib_path = project_path + r'/word-library'
    input_text_path = project_path + r'/input/content/sentiment_train'
    input_dish_path = word_lib_path + r'/dishName'
    input_location_path = word_lib_path + r'/locationName'
    input_dict_path = word_lib_path + r'/userdict.txt'
    stopwords_path = word_lib_path + r'/stopwords.txt'
    output_text_path = project_path + r'/output/sentiment_train'
    ltp_wordSegmentModel_path = project_path + r'/ltp_data_v3.4.0/cws.model'


    main()
