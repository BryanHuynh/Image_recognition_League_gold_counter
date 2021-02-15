import requests 
import json
import shutil

url = "http://ddragon.leagueoflegends.com/cdn/11.3.1/data/en_US/item.json"
data = requests.get(url).json()
d = {}

for id in data['data']:
    d[id] = ({
        'id'  : id,
        'name': data['data'][id]['name'],
        'gold': data['data'][id]['gold']['base']
    })

    img_url = "http://ddragon.leagueoflegends.com/cdn/11.3.1/img/item/"+id+".png"
    img_data = requests.get(img_url, stream=True)
    with open('images/'+id+'.png', 'wb') as out_file:
        shutil.copyfileobj(img_data.raw, out_file)
    del img_data



with open('data.json', 'w') as outfile:
    json.dump(d, outfile)

