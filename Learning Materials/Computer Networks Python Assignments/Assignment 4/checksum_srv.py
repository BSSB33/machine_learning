#checksum server

import socket
import sys
import select
import time

checksums = []
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip = sys.argv[1]
port = int(sys.argv[2])
address = (ip, port)
sock.bind(address)
sock.listen(1)
inputs = [sock]

while True:
	read, write, exc = select.select(inputs, [], [])
	for r in read:
		if r is sock:	
			cSock, cAddr = r.accept()
			inputs.append(cSock)
		else:
			#print(r)
			data = r.recv(1024)
			if data:
				data = data.decode()
				processsedData = data.split('|')
				if processsedData[0] == 'BE':
					checksums.append([processsedData[1], processsedData[2], processsedData[3], processsedData[4], time.time()])
					r.send(b'OK')
				elif processsedData[0] == 'KI':
					isEnd = False
					for i in checksums:
						if i[0] == processsedData[1] and (time.time()-float(i[4])) <= float(i[1]):
							length = i[2]
							checksum = i[3]
							isEnd = True
					if isEnd:
						r.send((str(len) + "|" + str(checksum)).encode())
					else:
						r.send(("0|").encode())
			else:
				r.close()
				inputs.remove(r)
sock.close()
