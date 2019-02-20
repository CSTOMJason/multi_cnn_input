from getdata import GetData
#create the model
import tensorflow as tf
import numpy as np

if __name__=="__main__":
    ob=GetData("./data.txt")
    data=ob.load_data(6000)
    print(len(data))
    print(type(data))
    #获得数据的word2index 和index2word，数据和数据的标签
    word2index,index2word,datas,labels=ob.word2index_index2word(data)
    cleardatas=ob.dropstopswords(datas,"./stopwords.txt")#获取干净的数据
    #lengthlist=[len(per) for per in cleardatas]
    #maxlength=max(lengthlist)#获取最大的文本
    re_datas,re_labels=ob.padding_datas(datas,word2index,labels)
    #数据的shuflle
    index=np.arange(len(re_datas))
    index=np.random.permutation(index)
    Datas=[]
    Labels=[]
    for perindex in index:
        Datas.append(re_datas[perindex])
        Labels.append(re_labels[perindex])
    
    Datas=np.array(Datas)
    Labels=np.array(Labels)
    train_datas=[]
    for per in Datas:
        temp=list(per)
        temp.append(0)
        train_datas.append(temp)
    train_datas=np.array(train_datas)
    print("the train_datas.shape is ",train_datas.shape)
    print("the Labels.shape is ",Labels.shape)
    
    
    with tf.name_scope("Input_layer"):
        input1=tf.placeholder(tf.int32,[None,347],name="Input1")
        input2=tf.placeholder(tf.int32,[None,347],name="Input2")
        inputy=tf.placeholder(tf.float32,[None,2],name="Inputy")
    lr=0.0001#学习率
    vocab_size=len(word2index.values())#词典的大小
    embedding_size=128
    #embedding layer

    with tf.name_scope("Embedding_layer"):
        with tf.device("/cpu:0"):
            W1=tf.Variable(tf.random_normal(shape=[vocab_size,embedding_size]),name="W1")
            W2=tf.Variable(tf.random_normal(shape=[vocab_size,embedding_size]),name="W2")
            embed_1=tf.nn.embedding_lookup(W1,input1,name="Embed_1")
            embed_2=tf.nn.embedding_lookup(W2,input2,name="Embed_2")
        embed_1_expand=tf.expand_dims(embed_1,axis=3)
        embed_2_expand=tf.expand_dims(embed_2,axis=3)
    print("the embed_1:\n",embed_1)
    print("the embed_2:\n",embed_2)
    num_filters=32
filter_sizes=[3,4,5]
pools=[]
#第一个输入卷积
with tf.name_scope("Conv1"):
    for i,filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv1_layer%s"%(str(i))):
            filter_shape=[filter_size,embedding_size,1,num_filters]
            W_embed_1=tf.Variable(tf.random_normal(shape=filter_shape),name="W_embed_1")
            b_embed_1=tf.Variable(tf.zeros(num_filters),name="b_embed_1")
            conv=tf.nn.conv2d(embed_1_expand,
                             W_embed_1,
                              strides=[1,2,2,1],
                              padding="SAME",
                              name="Conv"
                             )
            conv_ac=tf.nn.leaky_relu(tf.nn.bias_add(conv,b_embed_1),name="leak_relu")
            pool=tf.nn.max_pool(conv_ac,
                               ksize=[1,128-filter_size,1,1],
                               strides=[1,2,2,1],
                               padding="SAME",
                               name="pool")
            pools.append(pool)
    #第二个输入卷积
    with tf.name_scope("Conv2"):
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv2_layer%s"%(str(i))):
                filter_shape=[filter_size,embedding_size,1,num_filters]
                W_embed_2=tf.Variable(tf.random_normal(shape=filter_shape),name="W_embed_2")
                b_embed_2=tf.Variable(tf.zeros(num_filters),name="b_embed_2")
                conv=tf.nn.conv2d(embed_2_expand,
                             W_embed_2,
                              strides=[1,2,2,1],
                              padding="SAME",
                              name="Conv"
                                )
                conv_ac=tf.nn.leaky_relu(tf.nn.bias_add(conv,b_embed_2),name="leak_relu")
                    pool=tf.nn.max_pool(conv_ac,
                               ksize=[1,128-filter_size,1,1],
                               strides=[1,2,2,1],
                               padding="SAME",
                               name="pool")
                pools.append(pool)
    with tf.name_scope("Pools_concat"):
        pools_concat=tf.concat(pools,axis=3,name="Pools_concat")
    print("the pools_concat:\n",pools_concat)
    
    
    
    #多层卷积+全连接层
    #多层卷积第一层
    with tf.name_scope("M_conv1"):
        m_W1=tf.Variable(tf.random_normal(shape=[3,3,192,128]),name="m_W1")
        m_b1=tf.Variable(tf.zeros(128),name="m_b1")
        m_conv1=tf.nn.conv2d(pools_concat,
                        m_W1,
                        strides=[1,1,1,1],
                        padding="VALID",
                        name="m_conv1")
        m_conv1_ac=tf.nn.leaky_relu(tf.nn.bias_add(m_conv1,m_b1),name="leak_relu")
        m_h1=tf.nn.max_pool(m_conv1_ac,
                     ksize=[1,1,1,1],
                     strides=[1,1,1,1],
                     padding="VALID",
                     name="m_pool1")
    print("the m_h1:\n",m_h1)
    #多层卷积第二层
    with tf.name_scope("M_conv2"):
        m_W2=tf.Variable(tf.random_normal(shape=[4,4,128,256]),name="m_W2")
        m_b2=tf.Variable(tf.zeros(256),name="m_b2")
        m_conv2=tf.nn.conv2d(m_h1,
                         m_W2,
                        strides=[1,2,2,1],
                        padding="VALID",
                        name="m_conv2")
        m_conv2_ac=tf.nn.leaky_relu(tf.nn.bias_add(m_conv2,m_b2),name="leak_relu")
        m_h2=tf.nn.max_pool(m_conv2_ac,
                       ksize=[1,2,2,1],
                       strides=[1,2,2,1],
                       padding="VALID",
                       name="m_h2")
    print("the m_h2:\n",m_h2)
    #全连接层
    m_h2_flat=tf.reshape(m_h2,[-1,int(m_h2.shape[1])*int(m_h2.shape[2])*int(m_h2.shape[3])])
    with tf.name_scope("FC"):
        W_fc=tf.Variable(tf.random_normal(shape=[int(m_h2_flat.shape[1]),1024]),name="W_fc")
        b_fc=tf.Variable(tf.zeros(1024),name="b_fc")
        out_fc=tf.nn.leaky_relu(tf.matmul(m_h2_flat,W_fc)+b_fc,name="out_fc")
    print("out_fc:\n",out_fc)
    #drouput layer
    # with tf.name_scope("Dropout_layer"):
    #     out=tf.nn.dropout(
    
    #     )
    with tf.name_scope("Pred"):
        W_p=tf.Variable(tf.random_normal(shape=[1024,2]),name="W_p")
        b_p=tf.Variable(tf.zeros(2),name="b_p")
        pred=tf.nn.leaky_relu(tf.matmul(out_fc,W_p)+b_p,name="pred")
    print("the Pred:\n",pred)
    with tf.name_scope("LOSS"):
        loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=inputy),name="LOSS")
    with tf.name_scope("Train"):
        train=tf.train.AdamOptimizer(lr).minimize(loss)
    with tf.name_scope("Accuracy"):
        correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(inputy,1),name="Correct_pred")
        accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32),name="Accuracy")
    trainx=train_datas[:4800]
    trainy=Labels[:4800]
    testx=train_datas[4800:]
    testy=Labels[4800:]

    
    
    trainx1=[]
    trainx2=[]
    testx1=[]
    testx2=[]
    for per in trainx:
        trainx1.append(per[:347])
        trainx2.append(per[347:])
    for per in testx:
        testx1.append(per[:347])
        testx2.append(per[347:])
    trainx1,trainx2,testx1,testx2=np.array(trainx1),np.array(trainx2),np.array(testx1),np.array(testx2)
    print("testx1.shape:\n",testx1.shape)
    print("testx2.shape:\n",testx2.shape)
    print("trainx1.shape:\n",trainx1.shape)
    print("trainx2.shape:\n",trainx2.shape)
    
    
    
    
    batch_size=32
    num_batch=trainx.shape[1]//batch_size
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(10):
            for i in range(num_batch):
                x1,x2,y=trainx1[i*batch_size:(i+1)*batch_size],trainx2[i*batch_size:(i+1)*batch_size],trainy[i*batch_size:(i+1)*batch_size]
                _,los,acc=sess.run([train,loss,accuracy],feed_dict={input1:x1,input2:x2,inputy:y})
                print("%dth the loss is %f and the accuracy is %f"%(step,los,acc))
        _,los,acc=sess.run([train,loss,accuracy],feed_dict={input1:testx1[:120],input2:testx2[:120],inputy:testy[:120]})
        print("In the Test data:\n the loss is %f and the accuracy is %f"%(los,acc))
    

        
    
    
    
    