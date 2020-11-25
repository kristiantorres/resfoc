"""
Writes the EBCDIC header of a SEGY to a text file

@author: Joseph Jennings
@version: 2020.09.26
"""
import segyio
import re

def parse_text_header(segyfile):
  '''
  Format segy text header into a readable, clean dict
  '''
  raw_header = segyio.tools.wrap(segyfile.text[0])
  # Cut on C*int pattern
  cut_header = re.split(r'C\s?\d\d?', raw_header)[1::]
  # Remove end of line return
  text_header = [x.replace('\n', ' ') for x in cut_header]
  text_header[-1] = text_header[-1][:-2]
  # Format in dict
  clean_header = {}
  i = 1
  for item in text_header:
    key = "C" + str(i).rjust(2, '0')
    i += 1
    clean_header[key] = item
  return clean_header

#sgy = segyio.open("./segy/SW8102.rode_0001.segy",ignore_geometry=True)
sgy = segyio.open("./mig/Z3NAM1989E_Migration.sgy",ignore_geometry=True)

ebcdic_header = parse_text_header(sgy)

with open('./mig/EBCDIC.txt','w') as f:
  for key,val in ebcdic_header.items():
    f.write("%s %s\n"%(key,val))

