
# Migrates the data (fltdat.H)
fltimgextprcnew.H: scripts/migfaultext2.py
	python scripts/migfaultext2.py

# Computes the residual migration
bigresmig.H: scripts/resang.py
	python scripts/resang.py

# Converts to angle
stormang.H: scripts/toangle.py
	python scripts/resang.py
