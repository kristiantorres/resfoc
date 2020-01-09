import velocity.pySyntheticGen as pySyntheticGen
import velocity.pyHypercube as pyHypercube
import velocity.Hypercube as Hypercube
import velocity.pyGenericIO as pyGenericIO
import re
class geoModel:

	def __init__(self,**kw):
		"""Create a new geomodel
		     nx - (100) Number of samples in x
		     ox - (0.)  First sample in x
			 dx - (4.)  Sampling in x
			 ny - (100) Number of samples in y
			 oy - (0.)  First sample in y
			 dy - (4.)  Sampling in y
			 dz - (4.)  Sampling in z
			 basement - (4000.) Basement property
			 nbasement - (50) Initial number of samples in the basement
			 properties - (['velocity']) Properties to model"""
		self.io=pyGenericIO.ioModes([]).getDefaultIO()
		self.mParams={"nx":100,"ox":0.,"dx":4.,"ny":100,"oy":0,"dy":4.,
		    "dz":4,"basement":4000.,"nbasement":50,"properties":["velocity"]}
		for k,v in kw.items():
			self.mParams[k]=v
		self.axX=Hypercube.axis(n=self.mParams["nx"],o=self.mParams["ox"],d=self.mParams["dx"],
			label="X")
		self.axY=Hypercube.axis(n=self.mParams["ny"],o=self.mParams["oy"],d=self.mParams["dy"],
			label="Y")	
		self.axZ=Hypercube.axis(n=self.mParams["nbasement"],o=0.,d=self.mParams["dz"],label="Z")
		self.axes=[self.axZ.getCpp(),self.axX.getCpp(),self.axY.getCpp()]
		self.model=pySyntheticGen.model(self.io,self.axes,self.mParams["properties"],self.mParams["basement"])
		self.deposit_int={}
		self.deposit_float={}
		self.deposit_string={}
		self.deposit_bool={}
		self.depositT={"base_param":"string","band1":"float","band2":"float2",
		  "band3":"float","ratio":"float","var":"float","layer_rand":"float",
		  "layer":"float","prop":"float","dev_layer":"float","dev_pos":"float",
		  "thick":"int"}
		self.compact_int={}
		self.compact_float={}
		self.compact_string={}
		self.compact_bool={}
		self.compactT={"compact":"float"}
		self.erodeB_int={}
		self.erodeB_float={}
		self.erodeB_string={}
		self.erodeB_bool={}
		self.erodeBT={"center2":"float","center3":"float","width2":"float",
		  "width3":"float","depth":"float","fill_depth":"float",
		  "fill_prop":"float"}
		self.erodeF_int={}
		self.erodeF_float={}
		self.erodeF_string={}
		self.erodeF_bool={}
		self.erodeFT={"depth",".5"}
		self.erodeR_int={}
		self.erodeR_float={}
		self.erodeR_string={}
		self.erodeR_bool={}
		self.erodeRT={"start2":"float","start3":"float","dist":"float",
		  "azimuth":"float","fill_depth":"float","fill_prop":"float",
		  "nlevels":"int","wavelength":"float","waveamp":"float","thick":"float"}
		self.fault_int={}
		self.fault_float={}
		self.fault_string={}
		self.fault_bool={}
		self.faultT={"azimuth":"float","begx":"float","begy":"float","begz":"float",
		  "dz":"float","daz":"float","perp_die":"float","deltaTheta":"float",
		  "dist_die":"float","theta_die":"float","theta_shift":"float","dir":"float"}
		self.gaussian_int={}
		self.gaussian_float={}
		self.gaussian_string={}
		self.gaussian_bool={}
		self.gaussianT={"center2":"float","center3":"float","center1":"fault",
		  "vplus":"float","var":"float"}
		self.squish_int={}
		self.squish_float={}
		self.squish_bool={}
		self.squish_string={}
		self.squishT={"azimuth":"float","max":"float","wavelength":"float",
		  "random_inline":"float","random_crossline":"float"}
		self.implace_int={}
		self.implace_float={}
		self.implace_string={}
		self.implace_bool={}
		self.implaceT={"ntSteps":"int","emplace":"bool","conform":"bool","down_decrease":"bool",
		  "down_amount":"float","down_dist":"float","center1":"float","center2":"float",
		  "center3":"float","axis1":"float","axis2":"float","axis3":"float","azimuth":"float",
		  "pctRemove":"float","prop":"float"}
	def implace(self,**kw):
		"""Add feature to model
			emplace - [True] Whether or not emplace a body into the model
			prop    - [4500.] Value to set body
			center1,center2,center3 [.5] Relativel location of center of anomaly
			axis1,axis2,axis3 - [.3] Relative axes for anomaly
			azimuth - [0.] Rotation azimuth for body
			pctRemove - [30.] Percentage of points to remove
 			conform - [True] Conform model arroudn shape introduced
			down_decrease - [True] Decrease below anomaly
			down_dist - [0.] Distance down to change model
			ntSteps- [50] Number of time steps
			down_amount [0.] Down amount"""
		self.implace_int,self.implace_float,self.implace_string,self.implace_bool=self.parseParams(kw,self.implaceT,self.implace_int,self.implace_float,
		  	self.implace_string,self.implace_bool)
		event=pySyntheticGen.inplace(self.implace_int,self.implace_float,self.implace_string,
			self.implace_bool,self.mParams["properties"])
		event.doAction(self.model)
	def squish(self,**kw):
		"""Squish a model
			aziumth - [0.] Azimuth for squishing
			max - [50.] Maximum shift in z
			wavelength- - [1.] Wavlength scaling
			random_inline - [.5] Random inline
			random_crossline - [.5] random crossline"""
		self.squish_int,self.squish_float,self.squish_string,self.squish_bool=self.parseParams(kw,self.squishT,self.squish_int,self.squish_float,
		  	self.squish_string,self.squish_bool)
		event  =pySyntheticGen.squish(self.mParams["properties"],self.squish_int,self.squish_float,
		  	self.squish_string,self.squish_bool)
		
		event.doAction(self.model)

	def gaussian(self,**kw):
		"""Add a gaussian anomaly
			center2 - [.5] Relative position of anomaly axis2
			center1 - [.5] Relative position of anomaly axis1
			center3 - [.5] Relative position of anomaly axis3
			vplus   - [1.] Value of anomaly to add
			var     - [.1] Relative variance of anomaly"""
		self.gaussian_int,self.gaussian_float,self.gaussian_string,self.gaussian_bool=self.parseParams(kw,self.gaussianT,
		 	self.gaussian_int,self.gaussian_float,self.gaussian_string,
		 self.gaussian_bool)
		event=pySyntheticGen.gaussian(self.mParams["properties"],self.gaussian_int,self.gaussian_float,
			self.gaussian_string,self.gaussian_bool)
		event.doAction(self.model)

	def fault(self,**kw):
		"""Fault model
			azimuth - [0.] Azimuth of fault
			begx    - [.5] Relative location of the begining of fault x
			begy    - [.5] Relative location of the begining of fault y
			begz    - [.5] Relative location of the begining of fault z
			dz      - [0.] Distance away for the center of a circle in z
			daz     - [.01] Distance away in azimuth 
			perp_die- [0.1] Dieoff of fault in in perpdincular distance
			deltaTheta-[.1] Dieoff in theta away from the fault
			dist_die-  [0.] Distance dieoff of fault
			theta_die- [0.01] Distance dieoff in thetat
			theta_shift-[.1] Shift in thetat for fault
			dir - [.1] Direction of fault movement"""
		self.fault_int,self.fault_float,self.fault_string,self.fault_bool=self.parseParams(kw,self.faultT,self.fault_int,self.fault_float,
		  	self.fault_string,self.fault_bool)
		event=pySyntheticGen.fault(self.mParams["properties"],self.fault_int,self.fault_float,self.fault_string,
			self.fault_bool)
		event.doAction(self.model)

	def erodeRiver(self,**kw):
		"""Erode a river shape
			start2 - [.5] Position (relative to axis length) to start river
			start3 - [.0] Position (relative) to start river
			dist   - [1.4] Length (relative) of river
			azimuth - [0.] Angle for river
			fill_prop - [0.] Fill value for deposition for river chanel
			fill_depth - [0.] Fill dpeth for river chanel
			nlevels - [1] Number of river chanel bends to layout
			wavelength - [.01] Wavelenth multiplier for random river path
			waveamp - [.01] Wave ampitude multiplier
			thick - [.3] Thicknewss of river chanel"""
		self.erodeR_int,self.erodeR_float,self.erodeR_string,self.erodeR_bool=self.parseParams(kw,self.erodeRT,self.erodeR_int,self.erodeR_float,
		   	self.erodeR_string,self.erodeR_bool)
		event=pySyntheticGen.erodeRiver(self.mParams["properties"],self.erodeR_int,self.erodeR_float,
		   	self.erodeR_string,self.erodeR_bool)
		event.doAction(self.model)

	def erodeFlat(self,**kw):
		"""Erode a flat surface
			depth [.1] Fractional depth (axis 1) to slice off"""
		self.erodeF_int,self.erodeF_float,self.erodeF_string,self.erodeF_bool=self.parseParams(kw,self.erodeFT,self.erodeF_int,self.erodeF_float,
			self.erodeF_string,self.erodeF_bool)
		event=pySyntheticGen.erodeFloat(self.mParams["properties"],self.erodeF_int,self.erodeF_float,
		  self.erodeF_string,self.erodeF_bool)
		event.doAction(self.model)
	def erodeBowl(self,**kw):
		"""Erode a bowl shape
			center2 - [.5] Create a bowl fractional amount into model2
			center3 - [.5] Create a bowl fractional amount into model3
			width2  - [.01] Width of bowl fractional to length of axis 2
			width3  - [.01] Width of bowl fractional to length of axis 3
			depth   - [.01] Depth of bowl fractional  to length of axis 1
			fill_depth - [.01] Fill depth of bowl fractional to length of axis 1
			fill_prop - [.3] Fill value, dependent on model parameter"""
		self.erodeB_int,self.erodeB_float,self.erodeB_string,self.erodeB_bool=self.parseParams(kw,self.erodeBT,self.erodeB_int,self.erodeB_float,
		 	self.erodeB_string,self.erodeB_bool)
		event=pySyntheticGen.erodeBowl(self.mParams["properties"],self.erodeB_int,self.erodeB_float,self.erodeB_string,self.erodeB_bool)
		event.doAction(self.model)
	 
	def compact(self,**kw):
		"""Compact layers
			compact - [0.] Compact layers"""

		self.compact_int, self.compact_float, self.compact_string,self.compact_bool=self.parseParams(kw,self.compactT,
				self.compact_int, self.compact_float, self.compact_string,
				self.compact_bool)
		event=pySyntheticGen.compact(self.mParams["properties"],self.compact_int, self.compact_float, 
			self.compact_string,self.compact_bool)
		event.doAction(self.model)
	def parseParams(self,ks,typ,intM,floatM,stringM,boolM):
		"""Internal function to parse parameters"""
		for k,b in ks.items():
			for k1,v1 in typ.items():
				if re.compile(k1).search(k):
					if v1 == "int":
						intM[k]=b
					elif v1=="float":
						floatM[k]=b
					elif v1=="string":
						stringM[k]=b
					elif v1=="bool":
						boolM[k]=b
		return intM,floatM,stringM,boolM
	def deposit(self,**kw):
		"""Deposit a layer
			base_param - ["velocity"] Base param to base all other properties
			band1  - [.60] Bandpass parameter axis 1 property dependent vs.band1=
			band2  - [.05] Bandpass parameter axis 2 property dependent 
			band3  - [.05] Bandpass parameter axis 3 property dependent 
			ratio  - [.4]  Base ratio of property to main property
			var    - [.0]  Variance from main parameter
			layer_rand - [.5] Randomness variation within layer
			layer  - [9999.] Layer Base value
			prop  - [1.4]
			dev_layer - [0.]
			dev_pos - [0.]
			thick - [0.]"""
		self.deposit_int, self.deposit_float, self.deposit_string,self.deposit_bool=self.parseParams(kw,self.depositT,
				self.deposit_int, self.deposit_float, self.deposit_string,
				self.deposit_bool)
		event=pySyntheticGen.deposit(self.mParams["properties"],self.deposit_int, self.deposit_float,
				self.deposit_string, self.deposit_bool)
		event.doAction(self.model)
	def getMinMax(self,prop):
		  return self.model.getMin(prop),self.model.getMax(prop)
	def getHyper(self):
		return self.model.getHyper()
	def getProp(self,prop):
		"""Get model propertt"""
		return self.model.getProp(prop)
