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

