import os
import slack
import json

##############
# SALCK-API-TOKEN
##############

# 发送消息
# client=slack.WebClient(token='xoxb-399587329188-4151773040272-mYXBpuFN1c9mIfrFOYErLzS9')
# response=client.chat_postMessage(channel='#test',text="Hello world!")
# assert response["ok"]
# assert response["message"]["text"]=="Hello world!"

# 发送文件
client=slack.WebClient(token='xoxb-399587329188-4151773040272-mYXBpuFN1c9mIfrFOYErLzS9')
# response=client.files_upload(channels='#solar-flare-forecasting',file="./sun_flares.txt")
response=client.files_upload(channels='#test',file="./sun_flares.txt")

assert response["ok"]


# def pureimg(data1):
#         data1 = '[{"text": "", "image_url": "'+data1+'"}]'
#         data1 = [json.loads(data1[1:-1])]
#         return data1

#This function will make the image url to correct format.

# slacker = slack.WebClient(token='xoxe.xoxp-1-Mi0yLTM5OTU4NzMyOTE4OC0xNDAzODk3NzI1NDEzLTQxMjgwNDYwNDM4NDItNDEzMDQ5MjM0MjkzMi1iNWM0Y2ZjNmIzZDUzYzUwNzZiYWI1ZjliY2M5ZmI3Zjk2MjMwYjUwNDhkNDQ3ZWRkOTE3YTc0YmI1OTkxZmM3')

# payoff=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'filename.png')
#It gives cross OS compatibility on filepath.
#
# response=slacker.files_upload(channel='#theta',file=payoff)
# payoff=response['file']['permalink']

#First We upload the local file to Slack and fetch permalink.
#If you do not have any local file just put the external image URL in the payoff.

# response=slacker.chat_postMessage(channel='#channel_name', text="Sample Text", username='Bot name', attachments=pureimg(payoff), icon_emoji=':emoji:')

#Then, We post to Slack Channel as a bot!