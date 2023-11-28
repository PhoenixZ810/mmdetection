import requests
import json

headers = {'Content-Type': 'application/json'}

payload = {
    "prompt": "（在开始我们的对话前，请清空之前的对话）如何看待“996是福报？”",
    "appid": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjo5MywidXNlcm5hbWUiOiLotbXnv5TlrociLCJhcHBseV9hdCI6MTcwMTE0NDA4ODYwOSwiZXhwIjoxNzA5MDkyODg2ODcwfQ.5nRnd4kuHSzAgHat8Zc8sKJ3E94Bxq0my4c71SddSX4",
    "token": "",
}
response = requests.post(
    'http://ecs.sv.us.alles-apin.openxlab.org.cn/v1/claude/v1/text/chat_by_slack',
    headers=headers,
    data=json.dumps(payload),
)

print(response.status_code)

print(response.json())
