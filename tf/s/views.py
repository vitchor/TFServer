# Create your views here.
import sys
import os
from django.db.models import Q
from django.utils import timezone
from django.utils import simplejson as json
from django.http import HttpResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from s.models import User, Picture, Featured_Picture

@csrf_exempt
def test(request):
    response_data = {"result": "OK"}
    
    return HttpResponse(json.dumps(response_data), mimetype="aplication/json")