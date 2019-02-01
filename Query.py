import re
q=[]
import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import math
from collections import OrderedDict
#returns lemmatized query
def getQuery(path):
    f=open(path,'r')

    str=""
    x=f.readlines()

    for cont in x:
        clean = re.compile('<.*?>')
        cont = re.sub(clean, "", cont)
        #print("-----Removing Punctuations and alphanumerics----")
        cont = re.sub(r'[,-]', " ", cont)
        cont = re.sub(r'[^\w\s]', '', cont)
        #print("-----Removing numbers----")
        cont = re.sub(r'[0-9]+', ' ', cont)
        #print("-----Removing blank lines----")
    for line in x:

        if re.match('Q[0-9]+:\r\n',line)!=None:
            #print "match"
            q.append(str)
            str = ""

        else:
            str = str + line
    q.append(str)

    lemQuery=processQuery(q)

    return lemQuery
#remove stop words and lemmatize
def processQuery(q):
    wnl=WordNetLemmatizer()

    qr=[]
    words=[]
    lemQuery=[]
    for query in q:
        qr.append(query.strip('\r\n'))
    stopWords=removeStop()
    for qry in qr:
        x=qry.split()
        tag = pos_tag(x)

        for word,t in tag:
            if word not in stopWords:
                #print word

                tag1=get_wordnet_pos(t)
                word=wnl.lemmatize(word,tag1)
                #print word
                word=word.encode('utf-8')
                words.append(word)

        if words!=[]:
            lemQuery.append(words)
        words=[]

    #print lemQuery
    return  lemQuery


#get list of stop words
def removeStop():
    stop=open("/people/cs/s/sanda/cs6322/resourcesIR/stopwords")
    stopWords=[]
    for line in stop:
        stopWords.append(line.strip('\n'))
    return stopWords


from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#return query index and maxtf for each doc
def getDict(q):
    dict={}
    maxtf={}
    doc=0
    for query in q:
        doc+=1
        maxTf=0
        for term in query:
            if (term,doc) not in dict:
                dict[(term,doc)]=1
                if maxTf<1:
                    maxTf=1
            else:
                cnt=dict[(term,doc)]
                cnt+=1
                dict[(term,doc)]=cnt
                if maxTf<cnt:
                    maxTf=cnt
        maxtf[doc]=maxTf


    return dict,maxtf




#returns dictionary of index of collection
def readIndex(p):
    index={}

    f=open(p)
    i=f.readlines()
    x=0
    for line in i:
        postList = []
        ind=line.split(':')
        if len(ind)==2:
            term=ind[0]
            rest=ind[1].split('||')
            df=rest[0]
            no=2
            for docs in range(int(df)):
                postList.append(rest[no])
                no+=1
            index[term]=(df,postList)

    return index
#return query and doc wts with w1 and w2 weighing s cheme
def computeWeights(query,qDict,maxtf,dDict):
    qno=0
    qWeights1={}
    qWeights2 = {}
    dWeights1={}
    dWeights2 = {}
    qWts1={}
    qWts2={}
    dWts1={}
    dWts2={}
    normq1={}
    normq2={}
    normd1={}
    normd2={}
    qlen=0
    norm_qw1=0
    norm_qw2=0
    norm_dw1=0
    norm_dw2=0
    for term in dDict:
        (df, pl)=dDict[term]

    for q in query:
        qlen+=len(q)
    avgqlen=qlen/len(query)
    for q in query:
        qno+=1
        for term in q:
            mtf=maxtf[qno]
            if term in dDict:

                (df,pl)=dDict[term]
                for docs in range(int(df)):
                    docid, termfreq, doclen, max_tf=pl[docs].split(',')
                    wd1= (0.4 + 0.6 * math.log10 (float(termfreq) + 0.5) / math.log10(float(max_tf) + 1.0)) * (math.log10(1400/ int(df))/ math.log10(1400))
                    wd2=(0.4 + 0.6 * (float(termfreq) / (float(termfreq) + 0.5 + 1.5 *(int(doclen) / 859))) * math.log10 (1400 / int(df))/math.log10 (1400))
                    dWeights1[(term,docid)]=wd1
                    dWeights2[(term, docid)] = wd2

                    #norm_dw1+=wd1**2
                    if docid in normd1 and docid in normd2:
                        normd1[docid]+=wd1**2
                        normd2[docid]+=wd2**2
                    else:
                        normd1[docid]=wd1**2
                        normd2[docid] = wd2 ** 2


                if (term,qno) in qDict:
                    w1=(0.4 + 0.6 * math.log10 (qDict[(term,qno)] + 0.5) / math.log10(int(mtf) + 1.0)) * (math.log10(1400 / int(df))/ math.log10 (1400))
                    w2=(0.4 + 0.6 * (int(qDict[(term,qno)]) / int((qDict[(term,qno)]) + 0.5 + 1.5 *(len(q) / avgqlen))) * math.log10 (1400 / int(df))/math.log10 (1400))
                    qWeights1[(term,qno)]=w1
                    qWeights2[(term, qno)] = w2

                    norm_qw1 += w1 ** 2
                    norm_qw2 += w2 ** 2
        normq1[qno] =norm_qw1
        normq2[qno]=norm_qw2
        norm_qw1=0
        norm_qw2=0
    print normd1
    for k in qWeights1:
        (t,qn)=k
        qWeights1[(t,qn)]=qWeights1[(t,qn)]/math.sqrt(normq1[qno])
    for k in qWeights2:
        (t,qn)=k
        qWeights2[(t,qn)]=qWeights2[(t,qn)]/math.sqrt(normq2[qno])

    for k in dWeights1:
        (t, dd) = k
        if normd1[dd]!=0:
            dWeights1[(t, dd)] = dWeights1[(t, dd)] / math.sqrt(normd1[dd])
    for k in dWeights2:
        (t, dd) = k
        dWeights2[(t, dd)] = dWeights2[(t, dd)] / math.sqrt(normd2[dd])
    return qWeights1, qWeights2, dWeights1, dWeights2



#return each doc's score for each query and top5 scoring doc's for each query
def computeCosine(query,dDict,qwts,dwts):
    qno=0
    score={}
    top5={}

    for q in query:
        qno+=1
        for term in q:
            if term in dDict:

                (df,pl)=dDict[term]
                tempDict={}
                for docs in range(int(df)):
                    docid, termfreq, doclen, max_tf=pl[docs].split(',')
                    wtd=dwts[(term,docid)]
                    #wtd2=dwts2[(term,docid)]
                    wt=qwts[(term,qno)]
                    #wt2=qwts2[(term,qno)]
                    if docid in tempDict:
                        tempDict[docid]+=(wtd*wt)
                    else:
                        tempDict[docid]=(wtd*wt)

        tempDict=OrderedDict(sorted(tempDict.items(), key=lambda x: x[1],reverse=True))
        top5[qno]=tempDict.items()[0:5]
        if qno not in score:
            score[qno]=tempDict


    return score,top5

# return query vectors
def getVectors(qwts):
    vector={}
    tempVector = {}
    for qno in range(1,21):
        for (t,q) in qwts:
            if q==qno:
                tempVector[t]=qwts[(t,q)]
        vector[qno]=tempVector
        tempVector = {}
    return vector
#return doc vectors
def getDocVectors(top5,dwts):

    docVector={}
    for qno in top5:
        for (docid,score) in top5[qno]:
            docVectorterm = {}
            for (term,doc) in dwts:
                if doc==docid:
                    if term not in docVectorterm:
                        docVectorterm[term]=dwts[(term,doc)]

            docVector[int(docid)]=docVectorterm

    return docVector
# for top 5 docs of each query, compute and print rank, score, headline and external identifier
def getPrintDocVectorFull(top5,dwts,p):
    docVector = {}
    docWords = {}
    dc=""
    wnl=WordNetLemmatizer()
    for qno in top5:
        print "docs for query"+str(qno)+":"
        print "rank\tscore\tidentifier\theadline\t"
        rank=0
        HDLN={}
        tempVec = {}
        stopwo=removeStop()
        
        for (docid, score) in top5[qno]:
            words=[]
            rank+=1
            if int(len(str(docid)))<4:

                no=4-len(str(docid))
                for i in range(no):
                    dc=dc+'0'
                dc=dc+str(docid)
            if dc!='':
                f = open(p + "/cranfield" + dc, 'r')
                flag=0
            lines = f.readlines()

            for line in lines:

                if re.match('<TITLE>',line)!=None:
                    flag=1
                    headline = ""

                else:
                    if re.match('</TITLE>',line)==None and flag==1:

                        headline+=line

                    elif re.match('</TITLE>',line)!=None:

                        break


            if dc!='':
                identifier="cranfield"+dc
            else:
                identifier="cranfield"+docid

            print str(rank)+"\t"+str(score)+"\t"+identifier+"\t"+headline
            dc = ""
            tempVec[rank]=(score,identifier,headline.split('\n'))
        docVector[qno]=tempVec
    return docVector




index=readIndex("alem1.uncompressed")
query=getQuery("/people/cs/s/sanda/cs6322/hw3.queries")
dict,maxtf=getDict(query)
qwts1,qwts2,dwts1,dwts2=computeWeights(query,dict,maxtf,index)
score1,top51=computeCosine(query,index,qwts1,dwts1)
score2,top52=computeCosine(query,index,qwts2,dwts2)
vec1=getVectors(qwts1)
vec2=getVectors(qwts2)

docVec1=getDocVectors(top51,dwts1)
docVec2=getDocVectors(top52,dwts2)
dv1=OrderedDict(sorted(docVec1.items()))
dv2=OrderedDict(sorted(docVec2.items()))


print "Vector representation of the query using w1"
print vec1
print "Vector representation of query using w2"
print vec2
print "Top 5 documents for each query using w1"
print top51
print "Top 5 documents for each query using w2"
print top52
print "rank score, external identifier, headline for docs using w1"
dvf1=getPrintDocVectorFull(top51,dwts1,"/people/cs/s/sanda/cs6322/Cranfield")
print "rank score, external identifier, headline for docs using w2"
dvf2=getPrintDocVectorFull(top52,dwts2,"/people/cs/s/sanda/cs6322/Cranfield")
print "Vector representation of top 5 docs using w1"
print docVec1
print "Vector representation of top 5 docs using w2"
print docVec2


