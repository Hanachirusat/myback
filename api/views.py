import time
import datetime

from rest_framework.views import APIView
from rest_framework.response import Response
import numpy as np
from FaderatedLearning.classifier import classifier as cl
from django.shortcuts import render
from api.models import User, History


def describe_to_symptom(describe):
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
        '流泪': 0}
    for key in describe:
        for i in symptom.keys():
            if key == i:
                symptom[key] = 1
    for i in symptom.keys():
        X_symptom.append(symptom[i])
    X = np.asarray(X_symptom)
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
