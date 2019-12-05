from slacker import Slacker

### Set up slacker for status updates
with open('../neurodata-slackr.conf', 'r') as fp:
    slack_token = fp.readline().strip()
    slack_channel = fp.readline().strip()

slack = Slacker(slack_token)

slack.chat.post_message(slack_channel, '\n\n :partyparrot: Finished a run.')



'''
The *.conf file should look like


<slack legacy token>
<channel name>
'''
