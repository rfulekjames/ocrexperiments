import json


with open('docs/new_twenty_codes.json', 'r') as f:
    codes = f.read()

codes = json.loads(codes)


codes.sort()



with open('docs/new_twenty_codes.json', 'w') as f:
    f.write(json.dumps(codes))
