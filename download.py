import requests

url = "https://drive.usercontent.google.com/download?id=1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4&export=download&authuser=0&confirm=t&uuid=4f92ee41-cfda-4cf1-b9ac-6f6ec1a00d53&at=AKSUxGNwdaE6RrD485tpmTpJJGtf%3A1761218335326"
r = requests.get(url)
open("file.zip", "wb").write(r.content)


