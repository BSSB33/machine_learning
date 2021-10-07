# Netcopy client

import socket
import struct
import sys
import os
import zlib

# szerver kapcsolat

ip = sys.argv[1]
port = int(sys.argv[2])

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

address = (ip, port)
sock.connect(address)

checkIp = sys.argv[3]
checkPort = int(sys.argv[4])
checkSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
checkAddress = (checkIp, checkPort)

fId = sys.argv[5] # fájl id
fPath = sys.argv[6] # fájl elérési útvonala

# fájl kezelése

txt = ""
with open(fPath, 'r') as file:
	for i in file:
		txt += i
file.close() # beolvasás befejezése

# crc ellenőrzés
crc = hex(zlib.crc32((txt).encode('UTF-8')) % (1<<32) )

checkSock.connect(checkAddress)
length = len(crc)
data = "BE|" + fId + "|60|" + str(length) + "|" + str(crc)
checkSock.sendall(data.encode('UTF-8'))
checkSock.close()

# küldés a szervernek

with open(fPath, 'r') as fileToSend:
	#print("sending file: ")
	for i in fileToSend:
		sock.sendall(i.encode('UTF-8'))
	end = "end"
	print("")
	sock.sendall(end.encode('UTF-8'))
		
sock.close()