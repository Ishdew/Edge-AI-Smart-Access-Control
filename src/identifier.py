from pathlib import Path
import numpy as np
import face_recognition as fr

import base64, string, random
import cv2

class Identifier:
	def __init__(self):
		self.view = bytes()
		self.encodings = {}
		self.exit = False
		self.friendly_names = {}
		self.allowed = []
		self.trained = [] # Tracks if the user has completed multi-shot training


		print('loading known people')
		#load known people
		p = Path('people')
		if not (p.exists() and p.is_dir()):
			print(p,'does not exsist as a directory, aborting')
			exit()

		for fn in p.glob('*.jpg'):
			# Skip full frames so they don't corrupt the encodings
			if 'full_frame' in fn.name:
				continue

			name = fn.stem #filename without extensio
			img = cv2.imread(str(fn))
			print('>>',name)

			try:
				self.encodings[name] = fr.face_encodings(img)[0]
			except Exception as e:
				print('failed to load',fn)
				print('reason:')
				print(e)
	
		# weird pathlib syntax
		p = p / 'meta.txt'
	
		if not(p.exists() and p.is_file()):
			print(p,'does not exist as a file, empty or otherwise')
			exit()
		
		for line in p.open('r').readlines():
			line = line.strip()
			if line == '':
				continue

			parts = [x.strip() for x in line.split(',')]
			uid = parts[0]
			access = parts[1] == 'True'
			
			trained = False
			name = uid
			
			# Parse the new 4-column format while supporting the old 3-column format
			if len(parts) >= 3:
				if parts[2] in ['True', 'False']:
					trained = parts[2] == 'True'
					if len(parts) >= 4:
						name = ' '.join(parts[3:])
				else:
					name = ' '.join(parts[2:])
					
			if uid not in self.encodings.keys():
				continue
				
			if access: self.allowed.append(uid)
			if trained: self.trained.append(uid)
			self.friendly_names[uid] = name

		print('loaded data')

	def setView(self, view):
		self.view = view

	def quit(self):
		self.exit = True

	async def stream(self, response):
		import asyncio
		while True:
			r = b''.join([b'--frame\r\nContent-Type:image/jpeg\r\n\r\n', self.view, b'\r\n'])
			await response.send(r)
			await asyncio.sleep(0.05) # Caps stream at 20 FPS to prevent crashing

	def toggleAccess(self, uid):
		if not uid in self.encodings.keys():
			return 'unknown user'
	
		if uid in self.allowed:
			self.allowed.remove(uid)
		else:
			self.allowed.append(uid)

		self.saveMeta()

		return 'ok'

	def hasAccess(self, uid):
		if uid not in self.encodings.keys():
			return False
		if uid in self.allowed:
			return True
		return False

	def saveMeta(self, fn = 'people/meta.txt'):
		p = Path(fn)
		with p.open('w') as f:
			for user in self.encodings.keys():
				allowed = user in self.allowed
				trained = user in self.trained
				name = self.friendly_names.get(user, user)
				# Write the new format: uid, allowed, trained, name
				f.write(f'{user},{allowed},{trained},{name}\n')


	def addNew(self, thumbnail, encoding):
		# generate random uid 8 char long
		c = string.ascii_uppercase + string.ascii_lowercase + string.digits
		
		uid = ''.join(random.choice(c) for _ in range(8))
		while uid in self.encodings.keys(): 
			uid = ''.join(random.choice(c) for _ in range(8))

		self.encodings[uid] = encoding
		cv2.imwrite('people/{}.jpg'.format(uid),thumbnail)
		
		self.saveMeta() # <--- ADD THIS LINE TO FORCE TEXT FILE UPDATE

		return uid
	

	def setName(self, uid, name):
		if uid not in self.encodings.keys():
			return False
		if name:
			self.friendly_names[uid] = name
		else:
			del self.friendly_names[uid]
		self.saveMeta()
		return True
		
	def getNames(self):
		ret = []
		for uid in self.encodings.keys():
			display_name = self.friendly_names.get(uid, uid)
			ret.append({
				'uid' : uid,
				'name' : display_name, 
				'allowed' : uid in self.allowed,
				'trained' : uid in self.trained # Send training status to dashboard
			})
		return ret

	def delete(self,uid):
		if not uid in self.encodings.keys():
			return None

		del self.encodings[uid]
		p = Path('people/{}.jpg'.format(uid))
		if p.exists() and p.is_file():
			p.unlink()
		p_full = Path('peoplefullframe/{}_full_frame.jpg'.format(uid))
		if p_full.exists() and p_full.is_file(): p_full.unlink()
	
	def getImageLocation(self, uid):
		if not uid in self.encodings.keys():
			return None
		return 'people/{}.jpg'.format(uid)

	def getIDFromEncoding(self, encoding, difference=0.40):
	
		other_encodings = list(self.encodings.values())
		distances = fr.face_distance(other_encodings, encoding)
		
		if not any([d <= difference for d in distances]):
			#this is a new face that we haven't seen before
			# so save it
			print('no user found')
			return None

		most_similar = np.argmin(distances)

		uid = list(self.encodings.keys())[most_similar] 

		# Calculate and output debug statement
		match_accuracy = 1 - distances[most_similar]
		print(uid, ' user, with accuracy {:.1%}'.format(match_accuracy))
		
		# Only average the face if the user is ALREADY fully trained. 
		# This prevents bad angles from breaking the training process itself.
		if distances[most_similar] < 0.40 and uid in self.trained:
			self.encodings[uid] = np.average(
					[ encoding, self.encodings[uid] ],
					axis=0, weights=[1, 2]) 
		return uid
