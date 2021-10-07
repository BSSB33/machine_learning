import select, socket, sys, struct, random, time

class Server:
	def __init__(self, addr='localhost', port=10000, timeout=1):
		self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.server.setblocking(0)
		self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.server.bind((addr, port))
		self.server.listen(5)
		print('Started on: ' + str(addr) + ':' + str(port))
		self.inputs = [ self.server ]
		self.timeout = timeout
		self.packer = struct.Struct('s I')
		self.randomNumber = random.randint(0,100)
		self.recivedNumber = -1
		self.endGame = False

	def start(self):
		while self.inputs:
			try:
				readable, writable, exceptional = select.select(self.inputs, [], self.inputs, self.timeout)
				if not (readable or writable or exceptional):
					continue
				self.handleInputs(readable)
			except KeyboardInterrupt:
				print("Close the system")
				for c in self.inputs:
					c.close()
				self.inputs = []

	def handleInputs(self, readable):
		for sock in readable:
			if sock is self.server:
				self.handleNewConnection(sock)
			else:
				self.handleDataFromClient(sock)

	def handleNewConnection(self, sock):
		connection, client_address = sock.accept()
		connection.setblocking(0)
		self.inputs.append(connection)

	def handleDataFromClient(self, sock):
		data = sock.recv(1024)
		data = data.strip()
		if data:
			data = self.packer.unpack(data)
			op = data[0].decode()
			self.recivedNumber = data[1]
			if op is '=':
				op = '=='

			evaluate = eval(f'{self.randomNumber}{op}{self.recivedNumber}')
			print(f'({self.randomNumber} {op} {self.recivedNumber}) =>', evaluate)
			
			if not self.endGame:
				if op is '==' and evaluate:
					response = b'Y'
					self.endGame = True
				elif op is '==' and not evaluate:
					response = b'K'
				elif evaluate:
					response = b'I'
				else:
					#if self.randomNumber is self.recivedNumber:
					#	response = b'Y'
					#	self.endGame = True
					#else:
					response = b'N'
			else:
				response = b'V'
			
			msg = (response, 0)
			packed = self.packer.pack(*msg)
			sock.sendall(packed)
		else:
			print('closing ' + str(sock.getpeername()) + ' after reading no data')
			self.inputs.remove(sock)
			sock.close()


if __name__ == "__main__":
	try:
		addr = sys.argv[1]
		port = int(sys.argv[2])

		server = Server(addr, port)
		server.start()
	except:
		server = Server()
		server.start()