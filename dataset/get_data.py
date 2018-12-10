#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:11:52 2018
@author: salar
"""

import urllib.request as urllib2
import numpy as np

name = "00000124.png"


def save_image(longitude, latitude, name):
  
  longitude = str(longitude)
  latitude = str(latitude)
  
  api = "https://maps.googleapis.com/maps/api/staticmap?"
  center = "center="+longitude+","+latitude
  zoom = "zoom=20"
  size = "size=800x800"
  map_type = "maptype=satellite"
  api_key = "key=AIzaSyCQTd-Qkwwpn6tbEa8-jHNgQjedOyrjvQs"
  
  URL = api+center+"&"+zoom+"&"+size+"&"+map_type+"&"+api_key
  
  print(URL)
  
  urllib2.urlretrieve(URL, name)
  

add_plus_minus = np.random.uniform(low=0.0, high=0.013818, size=(100,2))

start_long = 40.718317
start_lat   = -73.998384

#save_image(1,2)

# for i in range(add_plus_minus.shape[0]):
num = 125
for i in range(1):
  longitude = start_long + add_plus_minus[i][0]
  latitude  = start_lat   + add_plus_minus[i][1]
  longitude = np.round_(longitude, decimals=6)
  latitude  = np.round_(latitude, decimals=6)
  
  name = '00000' + str(num) + '.png'
  num = num + 2
  
  print('Saving image for longitude=%f and latitude=%f' % (longitude, latitude))
  save_image(longitude,latitude, name)


  # from IPython.display import Image, display
  # display(Image(filename=name))
  
  
#  print('Continue?[y/n]')
#  cmd = input()
#  if cmd == 'y':
#    continue
#  else:
#    break

print('End of script')