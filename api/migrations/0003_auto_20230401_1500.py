# -*- coding: utf-8 -*-
# Generated by Django 1.11.7 on 2023-04-01 07:00
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_remove_user_sex'),
    ]

    operations = [
        migrations.AlterField(
            model_name='history',
            name='time',
            field=models.DateTimeField(),
        ),
    ]
