import time
import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
from FaderatedLearning.classifier import classifier as cl
from django.shortcuts import render
from api.models import User, History


def describe_to_symptom(describe):
    #定义症状描述的近义词
    synonym_symptom={
        '1': [''],
        '发热' : ['发热','发烧'],
        '头痛': ['头痛','头疼','脑壳疼'],
        '胸痛': ['胸痛','胸疼',''],
        '腹痛': ['腹痛','腹疼','肚子疼','肚子痛'],
        '咳嗽': ['咳嗽'],
        '疲倦': ['疲倦'],
        '心律失常': ['心律失常','心跳加快','心跳快'],
        '耳鸣': ['耳鸣'],
        '心悸': ['心悸','心慌'],
        '无力': ['无力','没精神','没力气'],
        '喉痛': ['喉痛','喉咙疼','喉疼','喉咙疼'],
        '咽痛': ['咽痛'],
        '恶心呕吐': ['恶心','呕吐','恶心呕吐'],
        '腹泻': ['腹泻','拉肚子'],
        '腹胀': ['腹胀','肚子胀'],
        '头晕': ['头晕'],
        '恶寒': ['恶寒','浑身发冷','浑身冷'],
        '无汗': ['无汗'],
        '鼻塞': ['鼻塞','鼻子不透气'],
        '黄鼻涕': ['黄鼻涕'],
        '脚底蜕皮': ['脚底蜕皮','脚底起皮'],
        '脚皮肤干裂': ['脚底皮肤干裂','脚皮肤干裂'],
        '脚底瘙痒': ['脚底瘙痒','脚底痒'],
        '血压高': ['血压高'],
        '眼疲劳': ['眼疲劳','眼睛累'],
        '眼异物感': ['眼异物感','眼睛里面感觉有东西'],
        '眼睛干': ['眼睛干'],
        '畏光': ['畏光','看到光眼睛疼','眼睛怕光'],
        '眼睛酸胀': ['眼睛酸胀','眼酸'],
        '眼睛发红': ['眼睛发红','眼珠红','眼里有红丝'],
        '眼睛刺痛': ['眼睛刺痛','眼睛疼','眼睛痛'],
        '视野中斑点状':['视野中斑点状','视野中心有东西'],
        '视野阴影': ['视野阴影','视野不清','视野模糊'],
        '视力下降': ['视力下降','视力突然下降'],
        '流泪': ['流泪','流眼泪','眼睛不由自主的流泪'],
    }

    X_symptom = []
    symptom = {
        '1': 1,
        '发热': 0,
        '头痛': 0,
        '胸痛': 0,
        '腹痛': 0,
        '咳嗽': 0,
        '疲倦': 0,
        '心律失常': 0,
        '耳鸣': 0,
        '心悸': 0,
        '无力': 0,
        '喉痛': 0,
        '咽痛': 0,
        '恶心呕吐': 0,
        '腹泻': 0,
        '腹胀': 0,
        '头晕': 0,
        '恶寒': 0,
        '无汗': 0,
        '鼻塞': 0,
        '黄鼻涕': 0,
        '脚底蜕皮': 0,
        '脚皮肤干裂': 0,
        '脚底瘙痒': 0,
        '血压高': 0,
        '眼疲劳': 0,
        '眼异物感': 0,
        '眼睛干': 0,
        '畏光': 0,
        '眼睛酸胀': 0,
        '眼睛发红': 0,
        '眼睛刺痛': 0,
        '视野中斑点状': 0,
        '视野阴影': 0,
        '视力下降': 0,
        '流泪': 0
    }
    #首先遍历用户传输的每个值，其次遍历规整化的症状描述的key（keys是描述，values是值默认为0)，如果用户的描述在key的近义词里面，则吧该key的值赋1
    for key in describe:
        for i in symptom.keys():
            print(f'i:{i}')
            if key in synonym_symptom[i]:
                symptom[i] = 1

    for i in symptom.keys():
        X_symptom.append(symptom[i])
    X = np.array(X_symptom)
    X = X.reshape(1, 36)
    return X


# 后端api
# 该类的url为http://127.0.0.1:8000/api/Result/
# 症状疾病预测
class ResultView(APIView):
    def post(self, request, *args, **kwargs):
        name = request.data['name']
        sex = request.data['sex']
        age = request.data['age']
        describe = request.data['describe']
        openid = request.data['openid']
        print(openid)
        print(describe)
        symptom = describe_to_symptom(describe.split())
        print(symptom)
        rate, disease = cl.Classifier(symptom)
        print(rate)
        reconmend = {
            '干眼症': '人工泪液',
            '脚气': '克霉唑乳膏',
            '高血压': '阿米洛利',
            '急性肠胃炎': '复方黄连素片',
            '飞蚊症': '氨碘肽滴眼液',
            '风寒感冒': '清开灵口服液、感冒清热颗粒',
            '风热感冒': '蓝芩口服液、银翘解毒颗粒'
        }
        result = {'rate': 0, 'disease': '', 'drug_name': '对不起，您输入的症状涉及到本系统未知领域'}
        if rate > '16':
            result['rate'] = rate
            result['disease'] = disease
        for key in reconmend.keys():
            if key == result['disease']:
                result['drug_name'] = reconmend[key]
        # return Response({"status":True})

        # 接下来是写历史记录
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        History.objects.create(name=name, sex=sex, age=age, describe=describe, rate=rate, disease=disease,
                               drug=result['drug_name'], time=now_time, openid=openid)
        return Response(result)


# 该类的url为http://127.0.0.1:8000/api/login/
# 用户登录
class loginview(APIView):
    # post固定写法
    def post(self, request, *args, **kwargs):
        # 带数据的请求数据挡在request中的data中，为json数据
        data = request.data
        print(data)
        # 查看是否是第一次登录，如果是就把用户信息添加进user表
        QuerySet = User.objects.filter(openid=data['openid'])
        if QuerySet.count() == 0:
            User.objects.create(username=data['username'], openid=data['openid'])

        # 返回的时候返回的是一个字典，Response会把字典封装成json数据
        result = {'status': '登陆成功', 'openid': ''}
        result['openid'] = data['openid']
        return Response(result)


class history(APIView):
    def post(self, request, *args, **kwargs):
        data = request.data
        result = []
        QuerySet = History.objects.filter(openid=data['openid'])
        if QuerySet.count() == 0:
            return Response(result)
        for item in QuerySet:
            mid = {'name': item.name, 'sex': item.sex, 'age': item.age, 'describe': item.describe, 'rate': item.rate,
                   'disease': item.disease, 'drug': item.drug, 'time': item.time}
            result.append(mid)
        return Response(result)


# http://127.0.0.1:8000/api/test/直接在浏览器中打开进行测试用的
class test(APIView):
    # post固定写法
    def get(self, request, *args, **kwargs):
        User.objects.create(username='1', openid='2')
        return render(request, "back_management.html")
