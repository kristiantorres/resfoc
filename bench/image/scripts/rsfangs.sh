#! /bin/bash

Transp plane=23 < fltimgextprc1.H > fltprepangrsf.H
/opt/RSF/bin/sfslant adj=y p0=-3 np=601 dp=0.01 < fltprepangrsf.H | /opt/RSF/bin/sfput label2=tan > flttan.H
< flttan.H /opt/RSF/bin/sftan2ang a0=-70 na=281 da=0.5 | /opt/RSF/bin/sfput label2=ang > fltangrsf.H
