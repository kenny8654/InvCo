import os
import sys
from invco import ROOT_DIR
sys.path.append(ROOT_DIR)
from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import *
from .res2lights import Res2lights, EncoderCNN, Model
from demo2 import Demo
from ingrs_vocab import Vocabulary
from args import get_parser

import requests

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
        final_output = {"output":output,"lights":lights,"img":str(request.FILES['image'])}

        content = RecipeSave(title=str(request.FILES['image']),content=final_output)
        content.save()
        return JsonResponse(final_output)

    # return HttpResponse('ok')

def getImages(request):
    path = settings.MEDIA_ROOT
    img_list = os.listdir(path + '/img')
    response = {"images":img_list}
    return HttpResponse(str(response))

@csrf_exempt
def getSavedRecipe(request):
    if request.method == 'POST' and request.POST.get("name", False):
        title = request.POST.get("name", False).split('/')[3].split('?')[0]
        if len(title.split('_')) == 2 :
            title = title.split('_')[0] + '.' + title.split('.')[1]
            recipes = RecipeSave.objects.filter(title=title)        
            return JsonResponse(eval(recipes[0].content))
        else:
            recipes = RecipeSave.objects.get(title=title)
            return JsonResponse(eval(recipes.content))
    else:
        return HttpResponse('error')
    
