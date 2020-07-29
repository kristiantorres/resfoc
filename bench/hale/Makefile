XPATH=/net/thing/scr2/joseph29/

###=========================
### Process entire dataset
###=========================

# convert to half offset
./dat/mymidpts.H: ./dat/allmidpts.HH
	Window < $< > $@
	echo "d2=0.067 o2=.132" >> $@

# Mute
midmute.H: ./dat/mymidpts.H
	sfmutter v0=1.4 < $< > $@ datapath=${XPATH}

# Velocity analysis
myscn.H: midmute.H
	sfvscan < $< semblance=y v0=1.4 nv=51 dv=0.025 > $@ \
		datapath=${XPATH}

scnmute.H: myscn.H
	sfmutter x0=1.5 v0=0.67 half=n < $< > $@ \
		datapath=${XPATH}

velrms.H: scnmute.H
	sfpick rect1=40 rect2=15 < $< > $@
	datapath=${XPATH}

# NMO and stack
nmo.H: velrms.H midmute.H
	sfnmo velocity=$< < midmute.H > $@ datapath=${XPATH}

stk.H: nmo.H
	sfstack < $< > $@ datapath=${XPATH}

# RMS to interval velocity
vint.H: velrms.H
	sfdix rect1=10 rect2=1 < $< > $@ datapath=${XPATH}

# Convert the velocity to depth
vintz.H: vint.H
	sftime2depth velocity=vint.H intime=y nz=900 z0=0 \
		dz=0.005 < $< > $@ datapath=${XPATH}

###=========================
### Process even midpoints
###=========================

a.H: ./dat/allmidpts.HH
	Window3d < ./dat/allmidpts.HH j3=2 >a.H

ahalf.H: a.H
	Window < $< >$@
	echo "d2=0.067 o2=0.132" >> $@

# Mute
amute.H: ahalf.H
	sfmutter v0=1.4 < $< > $@ datapath=${XPATH}

# Velocity analysis
ascn.H: amute.H
	sfvscan < $< semblance=y v0=1.4 nv=51 dv=0.025 > $@ \
		datapath=${XPATH}

###=========================
### Process odd midpoints
###=========================

b.H: ./dat/allmidpts.HH
	Window3d < ./dat/allmidpts.HH f3=1 j3=2 >b.H
	echo "o2=.331" >>b.H

bhalf.H: b.H
	Window < $< >$@
	echo "d2=0.067 o2=0.1655" >> $@

# Mute
bmute.H: bhalf.H
	sfmutter v0=1.4 < $< > $@ datapath=${XPATH}

# Velocity analysis
bscn.H: bmute.H
	sfvscan < $< semblance=y v0=1.4 nv=51 dv=0.025 > $@ \
		datapath=${XPATH}

###============================
### Combine for Dix inversion
###============================

velrmscomb.H: ascn.H bscn.H ./scripts/combvels.py
	python ./scripts/combvels.py

vinttcomb.H: velrmscomb.H
	sfdix rect1=10 rect2=1 < $< > $@ \
		datapath=${XPATH}

vintzcomb.H: vinttcomb.H
	sftime2depth velocity=$< intime=y nz=900 z0=0 \
		dz=0.005 < $< > $@ datapath=${XPATH}

###==========================
### Convert to shots
###==========================

threeD.H:  ./dat/allmidpts.HH
	Window3d < ./dat/allmidpts.HH j3=2 >a.H
	Window3d < ./dat/allmidpts.HH j3=2 >b.H f3=1
	echo o2=.331 >>b.H
	Create3d < a.H >a.3.H keyname1=offset keyname2=cmp_x
	Create3d < b.H >b.3.H keyname1=offset keyname2=cmp_x
	Scat3d a.3.H b.3.H >$@ axis=2

math.H: threeD.H
	Headermath < threeD.H >$@ key1=s_x eqn1="cmp_x-offset/2" key2=g_x eqn2="cmp_x+offset/2" \
		type1="scalar_float" type2="scalar_float"

sort.H: math.H
	Sort3d < math.H >$@ nkeys=2 key1=offset key2=s_x ng1=48 og1=.264 dg1=.067  \
		ng2=173 og2=5.36 dg2=.067 verb=1 verbose=1

shots.H: sort.H
	Infill3d < sort.H > shots.H

# Reshape gathers before migration
hale_shotflatbob.H: shots.H ./scripts/bobflat.py
	python ./scripts/bobflat.py

burn:
	Rm *.H
