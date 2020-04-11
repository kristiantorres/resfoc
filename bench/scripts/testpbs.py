import clusterhelp.pbs as pbs

cmd='echo "hello"'

pbs.write_script('example',cmd)


