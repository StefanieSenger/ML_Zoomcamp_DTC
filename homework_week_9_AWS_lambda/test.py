import requests

url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
#data = {'url': 'https://upload.wikimedia.org/wikipedia/commons/d/df/Smaug_par_David_Demaret.jpg'}
data = {'url': 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}

result = requests.post(url, json=data).json()
print(result)
