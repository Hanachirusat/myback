
from django.conf.urls import url,include
from django.contrib import admin
from api import views

urlpatterns = [
    #由上层url给出后寻找具体的url并且映射到具体的view函数
    #本例的上层url是 ../api/..
    url(r'^Result/', views.ResultView .as_view()),  #ResultView是views中的类名
    url(r'^login/', views.loginview .as_view()),
    url(r'^test/',views.test.as_view()),
    url(r'^history/',views.history.as_view()),
]
