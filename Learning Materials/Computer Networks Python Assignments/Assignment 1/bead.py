import subprocess
import sys
from datetime import date
import time
import json

#checks length of arguments
if len(sys.argv) < 2:
    print('Imput file required!')
    sys.exit()

#collects and process hosts
first_ten = subprocess.Popen(["head", "-10", sys.argv[1]], stdout=subprocess.PIPE)
data = first_ten.communicate()[0].decode('utf-8').strip('\n').split('\n')
last_ten = subprocess.Popen(["tail", "-10", sys.argv[1]], stdout=subprocess.PIPE)
data += last_ten.communicate()[0].decode('utf-8').strip('\n').split('\n')

hosts = []
traces = []
pings = []
today = date.today().strftime("%Y%m%d")

for host in data:
    hosts.append( host.split(',')[1] )

#calling procedures
i = 1
for host in hosts:
    print(str(i) + ". =======================================================================")
    print("==== Tracing: " + host + " ====")
    trace = subprocess.Popen(["traceroute", "-m", "30", host], stdout=subprocess.PIPE)
    traces.append(trace.communicate()[0].decode('utf-8').strip('\n'))
    print(traces[len(traces) - 1])
    print("==== Trace ENDED" + " ====")
    
    print("\n")
    
    print("==== Pinging: " + host + " ====")
    ping = subprocess.Popen(["ping", "-c", "10", host], stdout=subprocess.PIPE)
    pings.append(ping.communicate()[0].decode('utf-8').strip('\n'))
    print(pings[len(pings) - 1])
    print("==== Pinging Ended" + " ====")
    i = i + 1

#export
print("export: ======================================================================")
traceroutes={ "date": today, "system": sys.platform }
traceroutes["traces"]=[]
i = 0
for trace in traces:
    traceroutes["traces"].append({ "target:":  hosts[i], "output": trace})
    print(hosts[i])
    i = i + 1

pingsout={ "date": today, "system": sys.platform }
pingsout["pings"]=[]
i = 0
for ping in pings:
    pingsout["pings"].append({ "target:":  hosts[i], "output": ping})
    i = i + 1
	
with open('traceroute.json','w') as outfile:
    json.dump(traceroutes, outfile)

with open('ping.json','w') as outfile:
    json.dump(pingsout, outfile)