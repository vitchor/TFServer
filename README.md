TFServer
========

1. UBUNTU SETUP:
  - Install mysql
  - Install apache2 server/client
  - Install git
  - Install django
  - Install python-MysqlDB
  - Configure wsgi files

2. UBUNTU PYTHON MATH SETUP:
  - apt-get install python-numpy
  - apt-get install python-scipy
  - apt-get install python-pandas
  - apt-get install python-sympy
  - apt-get install python-sklearn
  - apt-get install cython
  - apt-get install python-setuptools
  - sudo easy_install -U scikit-image
  - sudo easy_install boto
  - sudo easy_install PIL

3. HOW TO FIX REQUIRE_DEBUG_FALSE BUG:
  - Remove any lines (and containers) containing require debug_false from both files:
    - /usr/lib/python2.7/dist-packages/django/conf/global_settings.py
    - [PROJECT_DIR]/tf/tf/settings.py

-------
TFServer: alternative web-server and WSGI handler (works better):
=================================================================

How to setup a django using nginx+uWSGI - full tutorial with tests on https://uwsgi.readthedocs.org/en/latest/tutorials/Django_and_nginx.html)

  1. If pip is still not available, install using:  
  ``` $ sudo apt-get install python-setuptools ```

  2. Download django:   
  ``` $ sudo pip install Django ```

  3. Install uwsgi:  
  ``` $ sudo pip install uwsgi ```

  4. Stop apache2 and uninstall it [if installed]:  
  ``` $ sudo service apache2 stop ```  
  ``` $ sudo apt-get remove apache2 ```

  5. Install nginx:  
  ``` $ sudo apt-get install nginx ```  
  ``` $ sudo service nginx start ```  

  6. Move to your project's path:  
  ``` cd /path/to/mysite/ ```  

  7. Create an uwsgi_params file on your mysite's root:  
  ``` $ sudo nano uwsgi_params ```  
    And paste the content from this file: https://github.com/nginx/nginx/blob/master/conf/uwsgi_params

  8. Create a file called mysite_nginx.conf and put this:

    ```
    # mysite_nginx.conf
    
    # the upstream component nginx needs to connect to
    upstream django {
        server unix:///path/to/your/mysite/mysite.sock; # for a file socket
        #server 127.0.0.1:8001; # for a web port socket (we'll use this first)
    }
    
    # configuration of the server
    server {
        # the port your site will be served on
        listen      8000;
        # the domain name it will serve for
        server_name .example.com; # substitute your machine's IP address or FQDN
        charset     utf-8;
    
        # max upload size
        client_max_body_size 75M;   # adjust to taste
    
        # Django media
        location /media  {
            alias /path/to/your/mysite/mysite/media;  # your Django project's media files - amend as required
        }
    
        location /static {
            alias /path/to/your/mysite/mysite/static; # your Django project's static files - amend as required
        }
    
        # Finally, send all non-media requests to the Django server.
        location / {
            uwsgi_pass  django;
            include     /path/to/your/mysite/uwsgi_params; # the uwsgi_params file you installed
        }
    }
    ```

  9. Symlink to this file from /etc/nginx/sites-enabled:  
  ``` $ sudo ln -s ~/path/to/your/mysite/mysite_nginx.conf /etc/nginx/sites-enabled/ ```

  10. Run this command (from inside your mysite's folder):  
  ``` $ uwsgi --socket mysite.sock --module mysite.wsgi --chmod-socket=666 ```

  11. Restart nginx:  
  ``` $ sudo service nginx restart ```

  - IMPORTANT NOTE: if you're doing this in a remote server (amazon EC2 or whatever), you will notice that once you've done the ```uwsgi``` command, you'll have your command line freezed. For this matter, replace step 10 with the ```screen``` tool as follows:  
  ``` $ screen ```  
  ``` $ uwsgi --socket mysite.sock --module mysite.wsgi --chmod-socket=666 ```  
  Press ``` ctrl + d ```  
  You can go back there anytime by typing ```$ screen -ls ``` and typing ```$ screen -r [PID_FROM_THE_SCREEN_YOU_WANT]```


