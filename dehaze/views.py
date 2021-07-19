from os import pipe
from django.shortcuts import render
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage
import sys
import os
from sample import fun

def home(request):
    return render(request,'home.html')

def dehaz(request):
    
    image=request.FILES['image']
    fs=FileSystemStorage()
    filename=fs.save(image.name,image)
    
    templateurl=fs.open(filename)
    fun(str(templateurl),str(filename))
    #dehazed = run([sys.executable,"E:\Mtech\Thesis\Webpage\Wavelet_Dehaze\sample.py",str(templateurl),str(filename)],shell=False,stdout=PIPE)
    
    templateurl = os.path.join("media",filename)
    rs =os.path.join("media","result.png")
    
    return render(request,'results.html',{'src':templateurl,'rs':rs})