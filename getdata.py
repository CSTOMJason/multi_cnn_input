import jieba
import re
import numpy as np
class GetData:
    def __init__(self,path):
        self.datapth=path
        self.vocab_size=None
    def load_data(self,number):#number表示要读取的数据多少个
        data=[]
        cnt=0
        with open(self.datapth,"r",encoding="utf-8") as f:
            while True:
                if cnt>=number:
                    break
                data.append(f.readline())
                cnt+=1
        data[0]=data[0].replace("\ufeff","")
        #清除中文文本中的标点符号和和其它的非中文字符
        tempdata=[]#临时的data
        for s in data:
            tempdata.append(self.remove_punctuation(s))
        del data
        return tempdata
    def remove_punctuation(self,s,strip_all=True):
        if strip_all:
            pattern=re.compile(r"[^a-zA-Z0-9\u4e00-\u9fa5]")
            s=pattern.sub("",s)
        else:
            punctuation = """！？｡＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"""
            re_punctuation = "[{}]+".format(punctuation)
            s= re.sub(re_punctuation, "",s)
        return s.strip()
    #获得word2index词汇到数字索引和index2word数字索引到词汇字典和对应的labels
    def word2index_index2word(self,data):
        counter=1
        labels=[]
        contents=[]
        word2index={}
        word2index["unk"]=0
        for s in data:
            labels.append(int(s[0]))#标签
            perwords=list(jieba.cut(s[1:],cut_all=False))#jieba分词精确模式
            contents.append(perwords)
            for word in perwords:

                if word not in word2index:
                    word2index[word]=counter
                    counter+=1
        index2word={index:word for word,index in word2index.items()}
        return word2index,index2word,contents,labels
    def dropstopswords(self,datas,stopwordspath):
        """drop the stopwords"""
        with open(stopwordspath,"r",encoding="utf-8") as f:
            stopwords=f.read().split("\n")
            stopwords[0]=stopwords[0].replace("\ufeff","")
        for i in range(len(datas)):
            for word in datas[i]:
                if word in stopwords:
                    datas[i].remove(word)
        return datas#清洗干净的数据
    def padding_datas(self,datas,word2index,labels):
        """padding the data with the same length"""
        lengthlist=[len(per) for per in datas]
        maxlength=max(lengthlist)#获取最大的文本
        train_datas=np.zeros([len(datas),maxlength])+0.1#使用非0的一个很小的数来填充(随机填充待实现)
        tempdatas=[]
        #获取句子对应的词典的索引
        for per in datas:
            temp=[]
            for word in per:
                temp.append(word2index.get(word))
            tempdatas.append(temp)
        for i,perdata in enumerate(tempdatas):
            train_datas[i][:len(perdata)]=perdata
        mid=labels.count(1)
        po_labels=np.array([[1,0] for i in range(mid)])
        ne_labels=np.array([[0,1] for i in range(mid,len(labels))])
       # return po_labels,ne_labels
        print(po_labels.shape)
        print(ne_labels.shape)
        self.vocab_size=len(word2index.values())
        return np.array(train_datas),np.concatenate([po_labels,ne_labels])













if __name__=="__main__":
    ob=GetData("F:\jupyter\python爬虫\data.txt")
    data=ob.load_data(4)
    print(len(data))
    print(type(data))
    #获得数据的word2index 和index2word，数据和数据的标签
    word2index,index2word,datas,labels=ob.word2index_index2word(data)
    cleardatas=dropstopswords(datas,"../stopwords.txt")#获取干净的数据
    #lengthlist=[len(per) for per in cleardatas]
    #maxlength=max(lengthlist)#获取最大的文本
    #获得数据集和对应的标签
    re_datas,re_labels=ob.padding_datas(datas,word2index,labels)








