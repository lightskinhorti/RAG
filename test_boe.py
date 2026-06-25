import requests
s = requests.Session()
s.headers.update({'User-Agent': 'RAG-BOE-Research/1.0', 'Accept': 'application/xml'})
r = s.get('https://www.boe.es/datosabiertos/api/boe/sumario/20250110', timeout=15)
print('Status:', r.status_code)
print('Content-Type:', r.headers.get('Content-Type', ''))
print(r.text[:1000])
