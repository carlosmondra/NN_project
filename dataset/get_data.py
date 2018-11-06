#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:11:52 2018

@author: salar
"""

import urllib

URL = "https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13&size=600x300&maptype=roadmap&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318&markers=color:red%7Clabel:C%7C40.718217,-73.998284&key="
api_key = "AIzaSyCQTd-Qkwwpn6tbEa8-jHNgQjedOyrjvQs"

#URL = "https://maps.googleapis.com/maps/api/staticmap?center=Brooklyn+Bridge,New+York,NY&zoom=13&size=600x300&maptype=roadmap
#&markers=color:blue%7Clabel:S%7C40.702147,-74.015794&markers=color:green%7Clabel:G%7C40.711614,-74.012318
#&markers=color:red%7Clabel:C%7C40.718217,-73.998284
#&key=AIzaSyCQTd-Qkwwpn6tbEa8-jHNgQjedOyrjvQs"

urllib.request.urlretrieve(URL+api_key, "00000001.png")