from slacker import Slacker
import socket

### Set up slacker for status updates
with open('neurodata-slackr.conf', 'r') as fp:
    slack_token = fp.readline().strip()
    slack_channel = fp.readline().strip()

slack = Slacker(slack_token)

host = socket.gethostname()
slack.chat.post_message(slack_channel, f'`{host}`\t Finished a run :partyparrot:')


slack.chat.post_message(slack_channel, f'`{host}`: Finished openml_d_{None}')

'''
The *.conf file should look like


<slack legacy token>
<channel name>
'''
