import socket, struct, sys, random, time

#midpoint = (first + last)//2
#if alist[midpoint] == item:
#	found = True
#else:
#	if item < alist[midpoint]:
#		last = midpoint-1
#	else:
#		first = midpoint+1

class Client:
	def __init__(self, serverAddress='localhost', serverPort=10000):
		self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.client.connect((serverAddress, serverPort))
		self.packer = struct.Struct('s I')
		
	def start(self):
		recived = None
		operators = [b'<', b'>', b'=']
		tipp, min, max = 50, 0, 100
		message = (b'>', tipp)

		while recived not in (b'Y', b'V'):
			operator = random.choice(operators)
			message = (operator, int(tipp))

			packed = self.packer.pack(*message)
			self.client.sendall(packed)

			recived_data = self.client.recv(1024)
			recived_data = self.packer.unpack(recived_data)
			recived = recived_data[0]

			if operator is b'>':
				if recived is b'I':
					min = tipp
				else:
					max = tipp
			if operator is b'<':
				if recived is b'I':
					max = tipp
				else:
					min = tipp
			tipp = ((min + max)/2)


			print(message, recived_data)
			time.sleep(1)

		print('Disconnected from server')		

if __name__ == "__main__":
	try:
		addr = sys.argv[1]
		port = int(sys.argv[2])
		
		client = Client(addr, port)
		client.start()
	except:
		client = Client()
		client.start()