from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse

from .models import *


def index(request):
    return render(request, 'index.html')

def uploadImg(request):
    if request.method == 'POST':
        img = ImgSave(img_url=request.FILES['image'])
        img.save()
    # save_path = '%s/image/%s'%(settings.MEDIA_ROOT,img.name)
    # with open(save_path, 'wb') as f:
    #     for content in img.chunks:
    #         f.write(content)
    # ImgSave.objectscreate(img_url='image/%s'%img.name)
    # # return render(request, 'index.html')
    return HttpResponse('ok')

