import json


with open('docs/twenty_codes-sorted.json', 'r') as f:
    codes = f.read()

approval_codes = json.loads(codes)


approval_codes.sort()


# for i, approval_code in enumerate(approval_codes):
#     if approval_code[:6] == 'Y0020 ':
#         approval_codes[i] = f'{approval_code[:5]}_{approval_code[6:]}'



with open('docs/twenty_codes-sorted.json', 'w') as f:
    f.write(json.dumps(approval_codes))
