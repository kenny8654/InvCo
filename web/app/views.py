import os
import sys
from invco import ROOT_DIR
sys.path.append(ROOT_DIR)
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from .models import *
from .res2lights import Res2lights, EncoderCNN, Model
from demo2 import Demo
from ingrs_vocab import Vocabulary
from args import get_parser

def index(request):
    return render(request, 'index.html')

def uploadImg(request):
    if request.method == 'POST':
        img = ImgSave(img_url=request.FILES['image'])
        img.save()
        r = Res2lights()
        lights = r.get_lights(str(request.FILES['image']))
        d = Demo()
        output = d.demo(str(request.FILES['image']),str(lights))
        final_output = {'output':output,'lights':lights}
        return HttpResponse(str(final_output))

    # return HttpResponse('ok')

def getImages(request):
    path = settings.MEDIA_ROOT
    img_list = os.listdir(path + '/img')
    response = {"images":img_list}
    return HttpResponse(str(response))

