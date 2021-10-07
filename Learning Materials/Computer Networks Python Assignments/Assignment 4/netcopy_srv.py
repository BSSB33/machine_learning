# Netcopy server

import socket
import struct
import sys
import os
import zlib
import select

ip = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
address = (ip, port)
sock.bind(address)

# kommunikáció
checkIp = sys.argv[3]
checkPort = int(sys.argv[4])
fId = sys.argv[5] # fájl id
fPath = sys.argv[6] # fájl elérési útvonala

sock.listen(1)

connection, clientAddress = sock.accept()
data = connection.recv(1024)

txt = ""
while data != "end":

	txt += str(data)
	data = connection.recv(1024)
	data = data.decode('UTF-8')

	with open(fPath, 'w') as file:
		file.write(txt)
	file.close()

connection.close()

#checksum
crc = hex(zlib.crc32((txt).encode('UTF-8')) % (1<<32) )
cConnection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
cAddress = (checkIp, checkPort)
cConnection.connect(cAddress)

cData = "KI|" + fId

length = len(crc)
cConnection.sendall(cData.encode('UTF-8'))
#print("data sent: " + str(cData.encode('UTF-8')))
checksum = cConnection.recv(length)
data = cConnection.recv(1024)
finalData = data.decode().split('|')

if finalData[0] == 0 or finalData[1] != checksum:
	print("CSUM CORRUPTED")
else:
	print("CSUM OK")
cConnection.close()