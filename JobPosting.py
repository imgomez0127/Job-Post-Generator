#!/usr/bin/python3
import requests
import json

# GET Request to get company information
headers = {"Authorization": "Bearer 128475ccf2cfadbe56a78fac4ea1a9291d0a9f617edd17d3dddd7395779c03f62391bf8e9057af957ff412fb77f871246911612131e8dfaa55201c244b236eda","Content-Type":"application/json"}
r = requests.get("https://hackicims.com/api/v1/companies/103/jobs", headers=headers)
# Do something with r.json() or r.text
# Status codes can be checked with r.status_code
print(r.json())
print(r.status_code)
# POST Request to make a job post named Software Developer for company 2002
# Status codes can be checked with r.status_code
