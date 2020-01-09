import velocity.pyHypercube as pyHypercube


class axis:
	def __init__(self,**kw):
		self.n=1
		self.o=0.
		self.d=1.
		self.label=""
		if "n" in kw:
			self.n=kw["n"]
		if "o" in kw:
			self.o=kw["o"]
		if "d" in kw:
			self.d=kw["d"]
		if "label" in kw:
			self.label=kw["label"]
		if "axis" in kw:
			self.n=kw["axis"].n
			self.o=kw["axis"].o
			self.d=kw["axis"].d
			self.label=kw["axis"].label

	def getCpp(self):
		return pyHypercube.axis(self.n, self.o, self.d, self.label)

class hypercube:
	def __init__(self,**kw):
		isSet=False
		self.axes=[]
		if "axes" in kw:
			for ax in kw["axes"]:
				self.axes.append(axis(n=ax.n,o=ax.o,d=ax.d,label=ax.label))
				isSet=True
		elif "ns" in kw:
			for n in kw["ns"]:
				self.axes.append(axis(n=n))
				isSet=True		
		if "os" in kw:
			for i in range(len(kw["os"]) -len(self.axes)):
				self.axes.append(axis(1))
			for i in range(len(os)):
				self.axes[i].o=kw["os"][i]
		if "ds" in kw:
			for i in range(len(kw["ds"]) -len(self.axes)):
				self.axes.append(axis(1))
			for i in range(len(kw["ds"])):
				self.axes[i].d=kw["ds"][i]
		if "labels" in kw:
			for i in range(len(kw["labels"]) -len(self.axes)):
				self.axes.append(axis(1))
			for i in range(len(os)):
				self.axes[i].label=kw["labels"][i]
		if "hypercube" in kw:
			for i in range(kw["hypercube"].getNdim()):
				a=axis(axis=kw["hypercube"].getAxis(i+1))
				self.axes.append(a)

	def getNdim(self):
		return len(self.axes)
	def getAxis(self,i):
		return self.axes[i-1]
	def getN123(self):
		n123=1
		for ax in self.axes:
			n123=n123*ax.n	
	def getCpp(self):
		ax1=self.axes[0].getCpp()
		if len(self.axes)>1:
			ax2=self.axes[1].getCpp()
			if len(self.axes)>2:
				ax3=self.axes[2].getCpp()
				if len(self.axes)>3:
					ax4=self.axes[3].getCpp()
					return pyHypercube.hypercube(ax1,ax2,ax3,ax4)
				else:
					return pyHypercube.hypercube(ax1,ax2,ax3)
			else:
				return pyHypercube.hypercube(ax1,ax2)
		else:
			return pyHypercube.hypercube(ax1)

	

