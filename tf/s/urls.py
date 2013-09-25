from django.conf.urls import patterns, include, url

urlpatterns = patterns('s.views',
    url(r'^test/$', 'test'),
    url(r'^uploas/$', 'upload'),
)