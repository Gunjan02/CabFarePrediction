
import requests

url = "http://localhost:5000/predict_api"
r = requests.post(url,json = {'passenger count':2,'day':10,'weekday':1,'hour':14,
                              'distance in km':10})
print(r.json())